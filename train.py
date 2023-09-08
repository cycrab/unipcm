import argparse
import json
import os
import random
import logging
import threading

#import jsonlines
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import LazyDataset
from tqdm import tqdm
from transformers import T5Tokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW
#import run_dialoglue

from args import parse_args, str2bool 
from t5_model import T5_model
import definitions

#RANK = int(os.environ['SLURM_PROCID'])
#RANK = int(os.environ['SLURM_PROCID'])

def get_optimizers(num_cases, model, hparams): # for pretraining
    no_decay = ["bias", "norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": hparams.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=hparams.lr)

    num_training_steps = num_cases*hparams.epoch_num // hparams.batch_size # gradient_accumulation_steps
    
    num_warmup_steps = hparams.warmup_steps if hparams.warmup_steps >= 0 else int(num_training_steps*hparams.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,\
        num_training_steps=num_training_steps) 
    return optimizer, scheduler

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=10,
                        help="random seed")

    parser.add_argument("--do_train", type=str2bool, default=False,
                        help="Whether to run trainning.")
    parser.add_argument("--do_test", type=str2bool, default=False,
                        help="Whether to run evaluation on the test dataset.")
    parser.add_argument("--do_infer", type=str2bool, default=False,
                        help="Whether to run inference on the test dataset.")
    parser.add_argument("--model_name", type=str, default='/data/nt12_hdd_gluster/myself/t5-base1',#checkpoints/checkpoint1/epoch0', 
                        help="t5-large,t5-1B")#'pretrain'
    parser.add_argument("--backbone", type=str, default='t5-base',
                        help="t5-large,t5-1B")#'pretrain'

    parser.add_argument("--batch_size", type=int, default=64, # 8
                        help="batch size")#'pretrain'
    parser.add_argument("--num_infer_batches", type=int, default=None,
                        help="The number of batches need to infer."
                        "Stay 'None': infer on entrie test dataset.")
    parser.add_argument("--max_seq_length", type=int, default=256, help="max length")
    parser.add_argument("--train_file", type=str, default='/data/nt12_hdd_gluster/myself/pre_train/train_encoded_50.jsonl', help="train file")
    parser.add_argument("--eval_file", type=str, default='/data/nt12_hdd_gluster/myself/pre_train/eval_encoded_50.jsonl', help="eval file")
    parser.add_argument("--num_tasks", type=int, default=11, help="task number to predict")

    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--epoch_num", type=int, default=15)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default=0, type=int, help="GPU device #")
    parser.add_argument("--num_device", default=1, type=int, help="GPU device #")
    parser.add_argument("--lr", type=float, default=3e-5,
                        help="learning rate")
    parser.add_argument("--warmup_steps", type=int, default=0, help="warmup")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="warmup ratio")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local rank for data parallel")

    parser.add_argument("--save_dir", type=str, default='/data/nt12_hdd_gluster/myself/checkpoint12',
                        help="save directory")
    hparams = parse_args(parser)
    hparams.use_gpu = torch.cuda.is_available() and hparams.local_rank >= 1    
    print(json.dumps(hparams, indent=2))
    if not os.path.exists(hparams.save_dir):
        os.makedirs(hparams.save_dir)
    hparams.save(os.path.join(hparams.save_dir, "hparams.json"))
    return hparams

def main(hparams):
    
    model = T5_model.from_pretrained(hparams.model_name, hparams) # hparams can get later via class functions

    def init_tokenizer(backbone):
        tokenizer = T5Tokenizer.from_pretrained(backbone)
        special_tokens = []
        """
        # add domains
        domains = definitions.ALL_DOMAINS + ["general"]
        for domain in sorted(domains):
            token = "[" + domain + "]"
            special_tokens.append(token)

        # add intents
        intents = list(set(chain(*definitions.DIALOG_ACTS.values())))
        for intent in sorted(intents):
            token = "[" + intent + "]"
            special_tokens.append(token)

        # add slots
        slots = list(set(definitions.ALL_INFSLOT + definitions.ALL_REQSLOT))

        for slot in sorted(slots):
            token = "[value_" + slot + "]"
            special_tokens.append(token)
        """
        special_tokens.extend(definitions.SPECIAL_TOKENS)
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        return tokenizer

    tokenizer = T5Tokenizer.from_pretrained(hparams.backbone)#hparams.model_name)
    model.resize_token_embeddings(len(tokenizer))

    hparams.pad_token_id = tokenizer.pad_token_id
    if hparams.local_rank != -1 :
        torch.distributed.init_process_group(backend="nccl", rank=hparams.local_rank)
    # construct model
    print("Total number of parameters in networks is {}".format(sum(x.numel() for x in model.parameters())))
    if hparams.local_rank == 0 :
        log_dir = os.path.join(hparams.save_dir,'tensorboard')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        tb_writer = SummaryWriter(log_dir=log_dir)

    def set_seed(seed):
        """ fix random seed """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        if hparams.local_rank > 0:
            torch.cuda.manual_seed_all(seed)

    # set seed
    set_seed(seed=hparams.seed)
    hparams.type = 'train' # type in train, eval
    train_set = LazyDataset(tokenizer,hparams)
    hparams.type = 'eval'
    eval_set = LazyDataset(tokenizer,hparams)

    optimizer, scheduler = get_optimizers(len(train_set), model, hparams)

    def collate_fn(self, tokenizer=tokenizer):
    # suit for list of dicts
    
        pad_id = tokenizer.pad_token_id
        pad_result = {}
        
        for key in ['input', 'output']:# add padding for input, ouput and attentions
            #np.array(
            #attention=len(encoded)*[1]
            #if  not isinstance(self[0][key],int): 
            max_len1 = max(len(input[key]) for input in self)
            max_len = min(hparams.max_seq_length, max_len1)
            pad_batch=np.ones((len(self), max_len))*pad_id  #-100
            pad_attention_batch=np.ones((len(self), max_len))*pad_id
            for idx, s in enumerate(self):
                #trunc = s[-max_len:]
                if len((s[key]))>max_len:
                    pad_batch[idx, :max_len] = np.array(s[key][-max_len:])
                    pad_attention_batch[idx, :max_len] = np.ones(max_len)
                else:
                    pad_batch[idx, :len(s[key])] = np.array(s[key])
                    pad_attention_batch[idx, :len(s[key])] = np.ones(len(s[key]))
            pad_result[(key)] = torch.from_numpy(pad_batch).long()
            pad_result[(key+'_attention')] = torch.from_numpy(pad_attention_batch).long()
            
        if 'task' in self[0]:
            pad_batch=np.ones(len(self))
            for idx, s in enumerate(self):
                pad_batch[idx] = s['task']
            pad_result['task'] = torch.from_numpy(pad_batch).long()
        return pad_result

    # set data paths and collate function

    # multi-gpu setting
    
    #if hparams.local_rank > 1 and torch.cuda.device_count() > 1:
    if hparams.local_rank != -1 :
        #device = torch.device('cuda', hparams.local_rank)
        torch.cuda.set_device(hparams.local_rank) # 
        model = model.cuda() # 
        model = DDP(model, device_ids=[hparams.local_rank], output_device=hparams.local_rank, find_unused_parameters=True) #find_unused_parameters
        train_sampler = DistributedSampler(train_set)
        eval_sampler = DistributedSampler(eval_set)
    
    else: 
        model.to(hparams.device)
        train_sampler = DistributedSampler(train_set) # need to be modified
        eval_sampler = DistributedSampler(eval_set)
        
    train_dataloader = DataLoader(dataset=train_set,
                                    batch_size=hparams.batch_size,
                                    collate_fn=collate_fn,
                                    #shuffle=True,
                                    sampler=train_sampler)
    val_dataloader = DataLoader(dataset=eval_set,
                                batch_size=hparams.batch_size,
                                collate_fn=collate_fn,
                                #shuffle=True,
                                sampler=eval_sampler)

    global_step = 0
    for epoch in range(hparams.epoch_num):
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0
        num_batches = 0
        oom_time = 0
        for batch in tqdm(train_dataloader):
            
            num_batches += 1
            global_step += 1

            # Transfer to gpu
            if torch.cuda.is_available():
                for key, val in batch.items():
                    if type(batch[key]) is list:
                        continue
                    batch[key] = batch[key].to(hparams.device if hparams.local_rank==-1 else hparams.local_rank)
            try:  # avoid OOM
                batch['output'][batch['output']==tokenizer.pad_token_id] = -100
                loss = model(input_ids=batch["input"], attention_mask=batch["input_attention"],labels=batch["output"], return_dict=False)[0]
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                batch_loss = loss.item()
                epoch_loss = epoch_loss + batch_loss
                if hparams.local_rank == 0 :
                    tb_writer.add_scalar('loss',batch_loss,global_step)

            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    #max_length = batch["input"].shape[1]
                    oom_time += 1
                    logging.info("WARNING: ran out of memory,times: {}, batch size: {}".format(#, max_len: {}
                        oom_time, hparams.batch_size))
                    torch.cuda.empty_cache()

        logging.info("Epoch loss: {}".format(epoch_loss / num_batches))
        print(epoch_loss / num_batches)
        # Evaluate and save checkpoint
        # score = evaluate(model, val_dataloader, hparams) # tokenizer,
        # logging.info("Eval loss: {}".format(score))
        # print(score)
        #torch.save(model.state_dict(), os.path.join(hparams.output_dir, "model.pt"))
        #torch.save(optimizer.state_dict(), os.path.join(hparams.output_dir, "optimizer.pt"))
        save_path = os.path.join(hparams.save_dir,'epoch'+str(epoch))
        if hparams.local_rank==0:
            model.module.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
        elif hparams.local_rank==-1:
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

def evaluate(model, val_dataloader, hparams): # tokenizer
    model.eval()
    epoch_loss = 0
    num_batches = 0
    global_step = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            num_batches += 1
            global_step += 1

            # Transfer to gpu
            if torch.cuda.is_available():
                for key, val in batch.items():
                    if type(batch[key]) is list:
                        continue
                    batch[key] = batch[key].to(hparams.device if hparams.local_rank==-1 else hparams.local_rank)
                
                batch['output'][batch['output']==hparams.pad_token_id] = -100
                loss = model(input_ids=batch["input"], attention_mask=batch["input_attention"],labels=batch["output"], return_dict=False)[0]
                epoch_loss += loss.item()
    return epoch_loss/num_batches

#parser = argparse.ArgumentParser()
#hparams = parse_args(parser)
#hparams.max_seq_length = 512
#tokenizer = T5Tokenizer.from_pretrained('t5-base')

if __name__ == "__main__":
    hparams = get_args()
    main(hparams)
    #load_dataset(tokenizer,hparams)