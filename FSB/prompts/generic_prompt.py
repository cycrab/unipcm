import torch
import math
import numpy as np
import json
from tqdm import tqdm
import logging
import copy
import random
from collections import defaultdict
import glob
import requests
from torch.utils.data import DataLoader
from transformers import AdamW
import torch.nn as nn

logging.getLogger('transformers.generation_utils').setLevel(logging.CRITICAL)

def load_prefix(tokenizer, shots_value, shot_converter, 
                file_shot, name_dataset, with_knowledge, 
                shot_separator="\n\n",sample_times=5):
    prefix_list = []
    for i in range(sample_times):
        shots = 0
        prefix_shot = {s:"" for s in shots_value}
        data = json.load(open(file_shot,"r"))
        random.Random(i).shuffle(data)
        prefix = ""
        for d in data:
            prefix += shot_converter(sample=d) + shot_separator
            shots += 1
            if shots in prefix_shot:
                prefix_shot[shots] = copy.copy(prefix)
        print(f"Loaded {name_dataset} {prefix_shot.keys()} shots for shuffle {i}!")
        prefix_list.append(prefix_shot)
    return prefix_list

def calculate_loss_and_accuracy(outputs, labels, pad_id): # for gpt
    # GPT2-chicahat/train.py
    lm_logits = outputs[0]

    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    # avg loss
    not_ignore = shift_labels.ne(pad_id)
    num_targets = not_ignore.long().sum().item()

    loss /= num_targets
    return loss

def finetune(sample, eval_sample, model, tokenizer, device, lr, gpt, epoch_num):
    NUM_EPOCHS = epoch_num # 5 # 7
    # if daily-dilog try finetuning 1 epoch
    if gpt:
        BATCH_SIZE = 4 # 2 for coqa # 6 for full data # 8 for 10%
    else:
        BATCH_SIZE = 8 # 8 # 4 for coqa
    if BATCH_SIZE == 4 or BATCH_SIZE == 8:
        gradient_accum = 4
    if BATCH_SIZE == 2:
        gradient_accum = 8
    else:
        gradient_accum = 2
    def collate_fn(self, tokenizer = tokenizer):
        # suit for list of dicts {'input':, 'output':}
        pad_result = {}
        if gpt:
            final = [(s['input']+s['output']+tokenizer.eos_token) for s in self]
            tokenized = tokenizer(final, return_tensors="pt", padding=True, truncation=True, max_length=512)
            pad_result['input_ids'] = tokenized.input_ids
            pad_result['input_mask'] = tokenized.attention_mask
            pad_result['output_ids'] = copy.deepcopy(tokenized.input_ids)
            for i in range(len(self)): # perform masking  for output
                input_length = len(tokenizer(self[i]['input'], max_length=512).input_ids)
                pad_result['output_ids'][i][:input_length]=tokenizer.pad_token_id
        elif 'blenderbot' in model.name_or_path:
            for key in ['input', 'output']:# same pad_id for all values
                data = [s[key] for s in self] # .replace('\n','').replace('Dialogue:','').replace('user','')
                tokenized = tokenizer(data, return_tensors="pt", padding=True, max_length=512)
                pad_result[(key + '_ids')] = tokenized.input_ids
                pad_result[(key + '_mask')] = tokenized.attention_mask
            #input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(self['input']))
            #input_ids = torch.LongTensor([input_ids])
            #data = [s['input'] for s in self]
            #input_ids = tokenizer.batch_encode_plus(data)
        else:
            for key in ['input', 'output']:# same pad_id for all values
                data = [s[key] for s in self]
                tokenized = tokenizer(data, return_tensors="pt", padding=True, max_length=512)
                pad_result[(key + '_ids')] = tokenized.input_ids
                pad_result[(key + '_mask')] = tokenized.attention_mask
            
        return pad_result

    optimizer = AdamW(model.parameters(), lr, eps=1e-8)

    train_dataloader = DataLoader(dataset=sample,
                                  batch_size=BATCH_SIZE,
                                  collate_fn=collate_fn,
                                  shuffle=True,
                                  pin_memory=True)
    eval_dataloader = DataLoader(dataset=eval_sample,
                                  batch_size=BATCH_SIZE,
                                  collate_fn=collate_fn,
                                  shuffle=False,
                                  pin_memory=True)
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        eval_loss = 0.0
        epoch_step = 0

        for batch in tqdm(train_dataloader):
            epoch_step = epoch_step + 1
            try:
                model.train()
                num_batches += 1
                if torch.cuda.is_available():
                    for key, val in batch.items():
                        if type(batch[key]) is list:
                            continue
                        batch[key] = batch[key].to(device)
                if ('blenderbot' in model.name_or_path) or gpt:
                    pass
                else:
                    batch["output_ids"][batch["output_ids"] == tokenizer.pad_token_id] = -100 # ignore pad_id in generation
                if gpt:
                    outputs = model(batch["input_ids"], attention_mask=batch["input_mask"])   
                    loss = calculate_loss_and_accuracy(outputs, labels=batch['output_ids'], pad_id=tokenizer.pad_token_id) # the input and output in gpt are the same
                else:
                    #tmp = batch["input_ids"].to('cpu').tolist()
                    #for t in tmp:
                    #    print(tokenizer.decode(t))
                    #print(batch["input_ids"].max())
                    #print(batch["output_ids"].max())
                    loss = model(input_ids=batch["input_ids"], attention_mask=batch["input_mask"], labels=batch["output_ids"], return_dict=False)[0]
                loss.backward()
                if epoch_step%gradient_accum==0:
                    optimizer.step()
                    optimizer.zero_grad()
                if loss.item()>-1000:  #!=float("nan"):
                    epoch_loss += loss.item()
                else:
                    pass
            except RuntimeError as exception:
                print(exception)
                torch.cuda.empty_cache()
                continue
        print(f"train loss:{epoch_loss}")
        if 'blenderbot' not in model.name_or_path:
            for batch in eval_dataloader:
                if torch.cuda.is_available():
                    for key, val in batch.items():
                        if type(batch[key]) is list:
                            continue
                        batch[key] = batch[key].to(device)
                if ('blenderbot' in model.name_or_path) or gpt:
                    pass
                else:
                    batch["output_ids"][batch["output_ids"] == tokenizer.pad_token_id] = -100 # ignore pad_id in generation
                model.eval()
                with torch.no_grad():
                    if gpt:
                        outputs = model(batch["input_ids"], attention_mask=batch["input_mask"])   
                        loss = calculate_loss_and_accuracy(outputs, labels=batch['output_ids'], pad_id=tokenizer.pad_token_id)
                    else:
                        loss = model(input_ids=batch["input_ids"], attention_mask=batch["input_mask"],labels=batch["output_ids"], return_dict=False)[0]
                    eval_loss += loss.item()
        print(f"eval loss:{eval_loss}")
    return epoch_loss

def load_samples(shot_converter, meta_type, max_number_turns, train_shot,
                file_shot, name_dataset, gpt,#with_knowledge, 
                label_proportion=10, augment=False): # 
    response_prompt = json.load(open('new_prompts/response_final.json','r'))
    samples = []
    eval = []
    eval_data = json.load(open(file_shot,"r"))
    train_data = json.load(open(train_shot,"r"))
    random.shuffle(train_data)
    train_num = int(len(train_data)*(label_proportion/100))
    data = train_data[:train_num]
    data.extend(eval_data)
    for i in range((train_num + len(eval_data))): # (sample_times):
        #shots = 0
        #random.Random(i).shuffle(data)
        #dialogue = data[0]
        dialogue = data[i]
        #if "data/smd/weather-" in file_shot or "data/smd/navigate-" in file_shot:
        #    prefix_shot[shot] = shot_converter(sample=data[0],with_knowledge=shot) 
        #else:
        if 'all_turns' in meta_type:
            if meta_type == "all_turns_category":
                temp =  {"personalities": [], "dialogue": []}
            else:
                temp =  {"meta": [], "dialogue": []}
        #if "img" in dialogue:   
        #    temp["img"] = dialogue["img"]
        #    temp["personalities"] = []
        else:
            if meta_type == "all":
                temp =  {"meta": dialogue["meta"], "dialogue": []}
            elif meta_type == "incremental" or meta_type == "none":
                temp =  {"meta": [], "dialogue": [], "KB": []}
            elif meta_type == "linear":
                temp =  {"meta": None, "dialogue": []}
            else:
                print("Choose a meta-type")

        for id_t, [user_utt, sys_utt] in enumerate(dialogue["dialogue"]):
            temp["dialogue"].append(["",""])
            if meta_type == "all_turns_category" or "img" in temp:
                temp["personalities"].append(dialogue["personalities"][id_t])

            #if meta_type == "all_turns_category" and id_t == 0:

            temp["dialogue"][-1][0] = user_utt

            #if sys_utt == "" or meta_type == "all_turns_category":
            
            no_prefix_query = shot_converter(sample=temp, gpt=gpt)
            temp["dialogue"][-1][1] = sys_utt
            temp["dialogue"] = temp["dialogue"][-max_number_turns:]
            if augment: # to be refined
                for p in response_prompt:
                    if isinstance(p,list):
                        samples.append({'input':p[0] + no_prefix_query + p[1], 'output':sys_utt})
                    else:
                        samples.append({'input':no_prefix_query + p, 'output':sys_utt})
            else:
                if sys_utt!='':
                    if i < train_num: 
                        samples.append({'input':no_prefix_query, 'output':sys_utt})
                    else:
                        eval.append({'input':no_prefix_query, 'output':sys_utt})
    print(f"Loaded {name_dataset}: {label_proportion} percent")
    return samples, eval

def load_prefix_by_category(tokenizer, shots_value, shot_converter, 
                file_shot, name_dataset, with_knowledge, 
                shot_separator="\n\n",sample_times=5):
    split = file_shot.replace(".json","")
    prefix_list = []
    for i in range(sample_times):
        prefix_shot_by_category = {}
        for file_shot_ in glob.glob(f"{split}/*_2.json"):
            shots = 0
            prefix_shot = {s:"" for s in shots_value}
            prefix = ""
            data = json.load(open(file_shot_,"r"))
            random.Random(i).shuffle(data)
            for d in data:
                prefix += shot_converter(sample=d) + shot_separator
                shots += 1
                if shots in prefix_shot:
                    prefix_shot[shots] = copy.copy(prefix)
            name_category = file_shot_.replace(".json","").replace(f"{split}/","")
            prefix_shot_by_category[name_category] = prefix_shot
        # print(prefix_shot_by_category[name_category])
        print(f"Loaded IC {len(prefix_shot_by_category.keys())} categories shots for shuffle {i}!")
        prefix_list.append(prefix_shot_by_category)

    return prefix_list



def compute_ppl(model, tokenizer, device, prefix, query, max_seq, image_chat=False, verbose=False, gpt=True): 
    if image_chat or not gpt:
        input_ids = tokenizer([prefix])
        label = torch.tensor(tokenizer([query])["input_ids"])
    else:
        if verbose:
            print(prefix+query)
        input_ids = tokenizer([prefix+query])
        if len(input_ids['input_ids'][0])>max_seq:
            input_ids['input_ids'][0] = input_ids['input_ids'][0][-max_seq:]
            input_ids["attention_mask"][0] = input_ids["attention_mask"][0][-max_seq:]

    total_input_len = len(input_ids["input_ids"][0])
    query_tok_len = len(tokenizer([query])['input_ids'][0])
    if gpt:
        label = torch.tensor([[tokenizer.pad_token_id]*(total_input_len-query_tok_len)+input_ids["input_ids"][0][-query_tok_len:]])
    input_ids["input_ids"] = torch.tensor(input_ids["input_ids"]).to(device)
    input_ids["attention_mask"] = torch.tensor(input_ids["attention_mask"]).to(device)
    label=label.to(device)
    with torch.no_grad():
        if gpt:
            outputs = model(input_ids["input_ids"], attention_mask=input_ids["attention_mask"])
            if input_ids["input_ids"].shape[1]==label.shape[1]:
                loss = calculate_loss_and_accuracy(outputs, labels=label, pad_id=tokenizer.pad_token_id)
                l = loss.item()
            else:
                l = 0.0
        else:
            outputs = model(input_ids['input_ids'], input_ids['attention_mask'], labels=label)
    if gpt:
        return l
    else:
        return outputs.loss.item()         
                    
def evalute_ppl(model, tokenizer, shot_converter, file_to_eval, 
                device, max_number_turns, with_knowledge, max_seq,
                prefix='', meta_type="all",verbose=False, gpt=True):
    if "all_turns" in meta_type:
        loss_list = []
        for dialogue in tqdm(json.load(open(file_to_eval,"r"))):
            if meta_type == "all_turns_category":
                temp =  {"personalities": [], "dialogue": []}
            else:
                temp =  {"meta": [], "dialogue": []}
    
            if "img" in dialogue:   
                temp["img"] = dialogue["img"]
                temp["personalities"] = []

            for id_t, [user_utt, sys_utt] in enumerate(dialogue["dialogue"]):
                temp["dialogue"].append(["",""])

                if meta_type == "all_turns_category" or "img" in temp:
                    temp["personalities"].append(dialogue["personalities"][id_t])

                ## prepare input prefix
                ## NOTICE: the last space is needed beacuse of GPT tokenizer 
                if meta_type == "all_turns_category" and id_t == 0:
                    pass
                else:
                    prefix_plus_dial_history = prefix + shot_converter(sample=temp)+" "
                    ppl = compute_ppl(model=model, tokenizer=tokenizer, 
                                    device=device, prefix=prefix_plus_dial_history, 
                                    query=user_utt, max_seq=max_seq, gpt=gpt)
                    if verbose:
                        print('----'*10)
                        print('----'*5+"PREFIX+DH"+'----'*5)
                        print('----'*10)
                        print(prefix_plus_dial_history)
                        print('----'*10)
                        print('----'*10)
                        print('----'*5+"GOLD"+'----'*5)
                        print('----'*10)
                        print(user_utt)
                        print(f"PPL: {math.exp(ppl)}")

                        print('----'*10)
                        input()
                    loss_list.append(ppl)

                temp["dialogue"][-1][0] = user_utt

                if sys_utt == "" or meta_type == "all_turns_category":
                    pass
                else:

                    prefix_plus_dial_history = prefix + shot_converter(sample=temp)+" "
                    ppl = compute_ppl(model=model, tokenizer=tokenizer, 
                                    device=device, prefix=prefix_plus_dial_history, 
                                    query=sys_utt, max_seq=max_seq, gpt=gpt)
                    if verbose:
                        print('----'*10)
                        print('----'*5+"PREFIX+DH"+'----'*5)
                        print('----'*10)
                        print(prefix_plus_dial_history)
                        print('----'*10)
                        print('----'*10)
                        print('----'*5+"GOLD"+'----'*5)
                        print('----'*10)
                        print(sys_utt)
                        print(f"PPL: {math.exp(ppl)}")
                        print('----'*10)
                        input()
                    loss_list.append(ppl)
                    

                # add gold utterance into sys_utt
                temp["dialogue"][-1][1] = sys_utt
                temp["dialogue"] = temp["dialogue"][-max_number_turns:]
            if verbose: break
        return math.exp(np.mean(loss_list))
    else:
        loss_list = []
        for dialogue in tqdm(json.load(open(file_to_eval,"r"))):
            if meta_type == "all":
                temp =  {"meta": dialogue["meta"], "dialogue": []}
            elif meta_type == "incremental" or meta_type == "none":
                temp =  {"meta": [], "dialogue": [], "KB": []}
            elif meta_type == "linear":
                temp =  {"meta": None, "dialogue": []}
            else:
                print("Choose a meta-type")

            for id_t, [user_utt, sys_utt] in enumerate(dialogue["dialogue"]):
                temp["dialogue"].append([user_utt,""])
                if meta_type == "incremental":
                    if "KB" in dialogue:
                        temp["KB"].append(dialogue['KB'][id_t])
                        temp["meta"] = dialogue["meta"]
                    else:
                        temp["meta"].append(dialogue['meta'][id_t])
                ## prepare input prefix
                ## NOTICE: the last space is needed beacuse of GPT tokenizer 
                prefix_plus_dial_history = prefix + shot_converter(sample=temp)+" "
                ppl = compute_ppl(model=model, tokenizer=tokenizer, 
                                device=device, prefix=prefix_plus_dial_history, 
                                query=sys_utt, max_seq=max_seq, gpt=gpt)


                if verbose:
                    print('----'*10)
                    print('----'*5+"PREFIX+DH"+'----'*5)
                    print('----'*10)
                    print(prefix_plus_dial_history)
                    print('----'*10)
                    print('----'*10)
                    print('----'*5+"GOLD"+'----'*5)
                    print('----'*10)
                    print(sys_utt)
                    print(f"PPL: {math.exp(ppl)}")
                    print('----'*10)
                    input()
                loss_list.append(ppl)

                # add gold utterance into sys_utt
                temp["dialogue"][-1][1] = sys_utt
                temp["dialogue"] = temp["dialogue"][-max_number_turns:]
                if meta_type == "incremental":
                    if "KB" in dialogue:
                        temp["KB"] = temp["KB"][-max_number_turns:]
                        assert len(temp["dialogue"]) == len(temp["KB"])
                    else:
                        temp["meta"] = temp["meta"][-max_number_turns:]
                        assert len(temp["dialogue"]) == len(temp["meta"])
            if verbose: break
        return math.exp(np.mean(loss_list))

def get_response_batch(model, tokenizer, device, do_sample, beam, gen_len, max_seq, eos_token_id, inputs):
    BATCH_SIZE = 8 # 16
    def collate_fn(self, tokenizer = tokenizer):
        # suit for list of dicts {'input':, 'output':}
        pad_result = {}
        for key in self[0]:
            if key =='input':# same pad_id for all values
                data = [s[key] for s in self]
                tokenized = tokenizer(data, return_tensors="pt", padding=True, max_length=512)
                pad_result[(key + '_ids')] = tokenized.input_ids
                pad_result[(key + '_mask')] = tokenized.attention_mask
            elif key=='output':
                pad_result[key] = [s[key] for s in self]
        pad_result['batch_size'] = len(self)
            
        return pad_result

    dev_dataloader = DataLoader(dataset=inputs,
                                  batch_size=BATCH_SIZE,
                                  collate_fn=collate_fn,
                                  shuffle=True,
                                  pin_memory=True)

    model.eval()
    labels = []
    generated = []
    with torch.no_grad():
        for batch in tqdm(dev_dataloader):

            if torch.cuda.is_available():
                for key, val in batch.items():
                    if (type(batch[key]) is list) or (type(batch[key]) is int):
                        continue
                    batch[key] = batch[key].to(device)

            output_token = model.generate(input_ids=batch["input_ids"], attention_mask=batch["input_mask"], eos_token_id=tokenizer.eos_token_id, max_length=gen_len, num_beams=beam)
            output_g = output_token.cpu().tolist()
            for j in range(batch['batch_size']):
                if 'blenderbot' in model.name_or_path:
                    generated.append(tokenizer.decode(output_g[j]).replace('__start__','').replace('__unk__',' ').replace('__end__','').replace('__null__',''))
                else:
                    generated.append(tokenizer.decode(output_g[j], skip_special_tokens=True))
                #if gpt
                #response = tokenizer.decode(output[0][input_len:])
                #response = response.split("\n")[0].strip()
            labels.extend(batch['output'])
    return (labels,generated)

def get_response(model, tokenizer, device, do_sample, beam, prefix_query, gen_len, max_seq, eos_token_id, multigpu, gpt, t5_input):
    input_ids = tokenizer(str(prefix_query), return_tensors='pt') if gpt else tokenizer(str(t5_input), return_tensors='pt')
    input_len = len(input_ids['input_ids'][0])
    if gpt:
        if input_len + gen_len > max_seq-200:
            print("WARNING: the prefix is too long, truncating it") 
            print(f"Tokenized length: {input_len}")
            token_to_remove = input_len + gen_len - (max_seq - 200)
            input_ids['input_ids'] = input_ids['input_ids'][:,token_to_remove:]
            input_len = len(input_ids['input_ids'][0])

            print(f"New Tokenized length: {input_len}")

    if multigpu: 
        with torch.no_grad():
            output = model.generate(
                **input_ids,
                do_sample=do_sample,
                max_length=input_len+gen_len if input_len+gen_len<max_seq else max_seq,
                eos_token_id=eos_token_id, # "\n"
                num_beams=beam,
                early_stopping=True,
            )
    else:
        with torch.no_grad():
            output = model.generate(
                input_ids = input_ids['input_ids'].to(device),
                do_sample=do_sample,
                max_length=(min(input_len+gen_len,max_seq) if gpt else gen_len),
                eos_token_id=tokenizer.eos_token_id, # (eos_token_id if gpt else tokenizer.eos_token_id), # "\n"
                num_beams=beam,
                early_stopping=True,
            )
    if gpt:
        response = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
        response = response.split("\n")[0].strip()
    else:
        response = tokenizer.decode(output[0], skip_special_tokens=True).replace('system:','').replace('Persona:','')
    return response
    

def get_prompt(dialogue,prefix,shot_converter):
    prompt = ""
    for shot in dialogue["shots"][:prefix]:
        prompt += shot_converter(sample=shot) + "\n\n"
    return prompt

def generate_response(model, tokenizer, shot_converter, file_to_eval, 
                      device, max_number_turns, with_knowledge, prefix='',
                      meta_type="all", gen_len=50, beam=1,max_seq=1024, 
                      eos_token_id=198, do_sample=False, multigpu=False, verbose=False, gpt=False):

    if "all_turns" in meta_type: # image-chat
        results = []
        for dialogue in tqdm(json.load(open(file_to_eval,"r"))):
            if meta_type == "all_turns_category":
                temp =  {"personalities": [], "dialogue": []}
            else:
                temp =  {"meta": [], "dialogue": []}
            if "img" in dialogue:   
                temp["img"] = dialogue["img"]
                temp["personalities"] = []
            
            res_temp =  {"meta":dialogue["meta"] , "dialogue": []}

            for id_t, [user_utt, sys_utt] in enumerate(dialogue["dialogue"]):
                temp["dialogue"].append(["",""])
                if meta_type == "all_turns_category" or "img" in temp:
                    temp["personalities"].append(dialogue["personalities"][id_t])

                if meta_type == "all_turns_category" and id_t == 0:
                    response_USR_A = ""
                    pass
                else:
                    prefix_query = prefix + shot_converter(sample=temp)
                    no_prefix_query = shot_converter(sample=temp)
                    response_USR_A = get_response(model, tokenizer, device, do_sample, beam, prefix_query, gen_len, max_seq, eos_token_id, multigpu, gpt, no_prefix_query)
                    if verbose:
                        print('----'*10)
                        print('----'*5+"PREFIX"+'----'*5)
                        print('----'*10)
                        print(prefix_query)
                        print('----'*10)
                        print('----'*10)
                        print('----'*5+"RESPONSE"+'----'*5)
                        print('----'*10)
                        print(response_USR_A)
                        print('----'*10)
                        input()

                temp["dialogue"][-1][0] = user_utt

                if sys_utt == "" or meta_type == "all_turns_category":
                    response_USR_B = ""
                    pass
                else:
                    prefix_query = prefix + shot_converter(sample=temp)
                    no_prefix_query = shot_converter(sample=temp)
                    response_USR_B = get_response(model, tokenizer, device, do_sample, beam, prefix_query, gen_len, max_seq, eos_token_id, multigpu, gpt, no_prefix_query)
                    if verbose:
                        print('----'*10)
                        print('----'*5+"PREFIX"+'----'*5)
                        print('----'*10)
                        print(prefix_query)
                        print('----'*10)
                        print('----'*10)
                        print('----'*5+"RESPONSE"+'----'*5)
                        print('----'*10)
                        print(response_USR_B)
                        print('----'*10)
                        input()
                temp["dialogue"][-1][1] = sys_utt

                res_temp["dialogue"].append([response_USR_A,response_USR_B])
                temp["dialogue"] = temp["dialogue"][-max_number_turns:]

            results.append(res_temp)
            if verbose: 
                break
        return results
    else:
        results = []
        inputs = []
        responses = []
        labels = []
        for dialogue in tqdm(json.load(open(file_to_eval,"r"))):
            if meta_type == "all":
                temp =  {"meta": dialogue["meta"], "dialogue": []}
            elif meta_type == "incremental" or meta_type == "none":
                temp =  {"meta": [], "dialogue": [], "KB": []}
            else:
                print("Choose a meta-type")

            res_temp =  {"meta": [], "dialogue": []}
            if "id" in dialogue:
                res_temp["id"] = dialogue["id"]
            for id_t, [user_utt, sys_utt] in enumerate(dialogue["dialogue"]):
                temp["dialogue"].append([user_utt,""])
                if meta_type == "incremental":
                    if "KB" in dialogue:
                        temp["KB"].append(dialogue['KB'][id_t])
                        temp["meta"] = dialogue["meta"]
                    else:
                        temp["meta"].append(dialogue['meta'][id_t])
                prefix_query = prefix + shot_converter(sample=temp, gpt=gpt)
                no_prefix_query = shot_converter(sample=temp, gpt=gpt)
                if sys_utt!='':
                    inputs.append({'input':no_prefix_query, 'output':sys_utt})
                
                temp["dialogue"][-1][1] = sys_utt
                temp["dialogue"] = temp["dialogue"][-max_number_turns:]
                if meta_type == "incremental":
                    if "KB" in dialogue:
                        temp["KB"] = temp["KB"][-max_number_turns:]
                        assert len(temp["dialogue"]) == len(temp["KB"])
                    else:
                        temp["meta"] = temp["meta"][-max_number_turns:]
                        assert len(temp["dialogue"]) == len(temp["meta"])
                if gpt:
                    r = get_response(model, tokenizer, device, do_sample, beam, no_prefix_query, gen_len, max_seq, eos_token_id, multigpu, gpt, no_prefix_query)
                    responses.append(r)
                    labels.append(sys_utt)
        if not gpt:
            response = get_response_batch(model, tokenizer, device, do_sample, beam, gen_len, max_seq, eos_token_id, inputs)
        else:
            response = (labels, responses)
        res_temp["dialogue"]=response
        results.append(res_temp)

        return results


def generate_response_dynamic(model, tokenizer, shot_converter, file_to_eval, 
                      prefix, device, max_number_turns, with_knowledge, 
                      meta_type="all", gen_len=50, beam=1,max_seq=1024, 
                      eos_token_id=198, do_sample=False, multigpu=False, verbose=False):

    if "all_turns" in meta_type:
        results = []
        for dialogue in tqdm(json.load(open(file_to_eval,"r"))):
            if meta_type == "all_turns_category":
                temp =  {"personalities": [], "dialogue": []}
            else:
                temp =  {"meta": [], "dialogue": []}
            if "img" in dialogue:   
                temp["img"] = dialogue["img"]
                temp["personalities"] = []
            
            res_temp =  {"meta":dialogue["meta"] , "dialogue": []}

            for id_t, [user_utt, sys_utt] in enumerate(dialogue["dialogue"]):
                temp["dialogue"].append(["",""])
                if meta_type == "all_turns_category" or "img" in temp:
                    temp["personalities"].append(dialogue["personalities"][id_t])

                if meta_type == "all_turns_category" and id_t == 0:
                    response_USR_A = ""
                    pass
                else:
                    prefix_query = prefix + shot_converter(sample=temp)
                    response_USR_A = get_response(model, tokenizer, device, do_sample, beam, prefix_query, gen_len, max_seq, eos_token_id, multigpu)
                    if verbose:
                        print('----'*10)
                        print('----'*5+"PREFIX"+'----'*5)
                        print('----'*10)
                        print(prefix_query)
                        print('----'*10)
                        print('----'*10)
                        print('----'*5+"RESPONSE"+'----'*5)
                        print('----'*10)
                        print(response_USR_A)
                        print('----'*10)
                        input()

                temp["dialogue"][-1][0] = user_utt

                if sys_utt == "" or meta_type == "all_turns_category":
                    response_USR_B = ""
                    pass
                else:
                    prefix_query = prefix + shot_converter(sample=temp)
                    response_USR_B = get_response(model, tokenizer, device, do_sample, beam, prefix_query, gen_len, max_seq, eos_token_id, multigpu)
                    if verbose:
                        print('----'*10)
                        print('----'*5+"PREFIX"+'----'*5)
                        print('----'*10)
                        print(prefix_query)
                        print('----'*10)
                        print('----'*10)
                        print('----'*5+"RESPONSE"+'----'*5)
                        print('----'*10)
                        print(response_USR_B)
                        print('----'*10)
                        input()
                temp["dialogue"][-1][1] = sys_utt

                res_temp["dialogue"].append([response_USR_A,response_USR_B])
                temp["dialogue"] = temp["dialogue"][-max_number_turns:]

            results.append(res_temp)
            if verbose: 
                break
        return results
    else:
        results = []
        for dialogue in tqdm(json.load(open(file_to_eval,"r"))):
            if meta_type == "all":
                temp =  {"meta": dialogue["meta"], "dialogue": []}
            elif meta_type == "incremental" or meta_type == "none":
                temp =  {"meta": [], "dialogue": [], "KB": []}
            else:
                print("Choose a meta-type")

            res_temp =  {"meta": [], "dialogue": []}
            if "id" in dialogue:
                res_temp["id"] = dialogue["id"]
            for id_t, [user_utt, sys_utt] in enumerate(dialogue["dialogue"]):
                temp["dialogue"].append([user_utt,""])
                if meta_type == "incremental":
                    if "KB" in dialogue:
                        temp["KB"].append(dialogue['KB'][id_t])
                        temp["meta"] = dialogue["meta"]
                    else:
                        temp["meta"].append(dialogue['meta'][id_t])
                prefix_query = prefix + shot_converter(sample=temp)
                if verbose:
                    print('----'*10)
                    print('----'*5+"PREFIX"+'----'*5)
                    print('----'*10)
                    print(prefix_query)
                    print('----'*10)

                response = get_response(model, tokenizer, device, do_sample, beam, prefix_query, gen_len, max_seq, eos_token_id, multigpu)
                res_temp["dialogue"].append([response])
                if verbose:
                    print('----'*10)
                    print('----'*5+"RESPONSE"+'----'*5)
                    print('----'*10)
                    print(response)
                    print('----'*10)
                    input()
                temp["dialogue"][-1][1] = sys_utt
                temp["dialogue"] = temp["dialogue"][-max_number_turns:]
                if meta_type == "incremental":
                    if "KB" in dialogue:
                        temp["KB"] = temp["KB"][-max_number_turns:]
                        assert len(temp["dialogue"]) == len(temp["KB"])
                    else:
                        temp["meta"] = temp["meta"][-max_number_turns:]
                        assert len(temp["dialogue"]) == len(temp["meta"])
            results.append(res_temp)
            if verbose:
                break
        return results



def evalute_prompt_prob(model, tokenizer, shot_converter, file_to_eval, 
                prefix, device, max_number_turns, with_knowledge, max_seq,
                meta_type="all",verbose=False, max_shot=1, repetition=0):

    loss_list = []
    id_dial = 0
    for dialogue in tqdm(json.load(open(file_to_eval,"r"))):
        id_dial += 1
        if id_dial == 101: break
        temp = {"dialogue": []}
        for id_t, [user_utt, sys_utt] in enumerate(dialogue["dialogue"]):
            temp["dialogue"].append([user_utt,""])

            prompt_ppl = defaultdict()
            for name, prompt in prefix.items():
                query = shot_converter(sample=temp)
                ppl = compute_ppl(model=model, tokenizer=tokenizer, 
                                device=device, prefix=prompt[repetition][max_shot] + " ", 
                                query=query, 
                                max_seq=max_seq)

                prompt_ppl[name] = math.exp(ppl)


            loss_list.append(prompt_ppl)
            # add gold utterance into sys_utt
            temp["dialogue"][-1][1] = sys_utt
            temp["dialogue"] = temp["dialogue"][-max_number_turns:]

        if verbose: break
    return loss_list


def select_prompt_interactive(model, tokenizer, shot_converter, dialogue, 
                prompt_dict, device, max_seq, max_shot=1, sample=False):
    temp = {}
    temp["dialogue"] = dialogue["dialogue"][-2:]
    query = shot_converter(sample=temp, with_knowledge=None)
    prompt_ppl = defaultdict()
    for name, prompt in prompt_dict.items():
        ppl = compute_ppl(model=model, tokenizer=tokenizer, 
                        device=device, prefix=prompt[max_shot], 
                        query=query,  max_seq=max_seq, verbose=False)
        prompt_ppl[name] = math.exp(ppl)
    if sample:
        sum_val = sum(prompt_ppl.values())
        prob_dict = {}
        for k, v in prompt_ppl.items():
            prob_dict[k] = sum_val-v
        return random.choices(list(prob_dict.keys()), weights=prob_dict.values(), k=1)[0]
    else:
        return min(prompt_ppl, key=prompt_ppl.get)


def generate_response_interactive(model, tokenizer, shot_converter, dialogue, 
                      prefix, device, with_knowledge, 
                      meta_type="all", gen_len=50, beam=1,max_seq=1024, 
                      eos_token_id=198, do_sample=False, multigpu=False, 
                      api=False, api_key=""):


    prefix_query = prefix + shot_converter(dialogue, with_knowledge)
    input_ids = tokenizer(str(prefix_query), return_tensors='pt')
    input_len = len(input_ids['input_ids'][0])

    if multigpu: 
        with torch.no_grad():
            output = model.generate(
                **input_ids,
                do_sample=do_sample,
                max_length=input_len+gen_len if input_len+gen_len<max_seq else max_seq,
                eos_token_id=eos_token_id, # "\n"
                num_beams=beam,
                early_stopping=True,
                top_p=0.9
            )
    elif api:
        response = requests.post(
            "https://api.ai21.com/studio/v1/j1-jumbo/complete",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "prompt": prefix_query, 
                "numResults": 1, 
                "maxTokens": input_len+gen_len if input_len+gen_len<max_seq else max_seq, 
                "stopSequences": ["\n"],
                "topP": 0.9
            }
        )
        json_data = json.loads(response.text)
        output = json_data['completions'][0]['data']['text']
        return output.split("\n")[0].strip()
    else:
        with torch.no_grad():
            output = model.generate(
                input_ids = input_ids['input_ids'].to(device),
                do_sample=do_sample,
                max_length=input_len+gen_len if input_len+gen_len<max_seq else max_seq,
                eos_token_id=eos_token_id, # "\n"
                num_beams=beam,
                early_stopping=True,
                top_p=0.9
            )
    response = tokenizer.decode(output[0][input_len:])
    response = response.split("\n")[0].strip()
    return response