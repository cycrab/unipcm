import argparse
import json
import logging
from tokenize import Ignore
import numpy as np
import os
import random
import torch

from typing import Any
from typing import Dict
from typing import TextIO
from typing import Tuple

from collections import Counter, defaultdict
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import T5Tokenizer#BertWordPieceTokenizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tqdm import trange
from transformers import AdamW

from constants import SPECIAL_TOKENS
from data_readers import IntentDataset, SlotDataset, TOPDataset
from bert_models import IntentT5Model, SlotT5Model, ExampleIntentBertModel,  JointSlotIntentT5Model, T5Pretrain
from transformers import T5ForConditionalGeneration

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
LOGGER = logging.getLogger(__name__)


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--test_data_path", type=str)
    parser.add_argument("--val_data_path", type=str, default='')
    parser.add_argument("--mlm_data_path", type=str, default='')
    parser.add_argument("--token_vocab_path", type=str)
    parser.add_argument("--output_dir", type=str, default='')
    parser.add_argument("--model_name_or_path", type=str, default="t5-base")
    parser.add_argument("--task", type=str, choices=["intent", "slot", "top"])
    parser.add_argument("--dump_outputs", action="store_true")
    parser.add_argument("--mlm_pre", action="store_true")
    parser.add_argument("--mlm_during", action="store_true")
    parser.add_argument("--example", action="store_true")
    parser.add_argument("--use_observers", action="store_true")
    parser.add_argument("--generation", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=50)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--do_lowercase", action="store_true")

    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--device", default=0, type=int, help="GPU device #")  ##0
    parser.add_argument("--max_grad_norm", default=-1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", type=list, default=['file1'])  # 'file1'
    # ['user: ','What is the user intent?']
    # ['file1']
    # settings for pet
    parser.add_argument("--prepet", action="store_true")
    parser.add_argument("--petgen", action="store_true")
    parser.add_argument("--pet_data_path", type=str)
    parser.add_argument("--pet_final", type=str, default='')

    parser.add_argument("--vote", action="store_true")
    return parser.parse_args()


def retrieve_examples(dataset, labels, inds, task, num=None, cache=defaultdict(list)):
    if num is None and labels is not None:
        num = len(labels) * 2

    assert task == "intent", "Example-driven may only be used with intent prediction"

    if len(cache) == 0:
        # Populate cache
        for i, example in enumerate(dataset):
            cache[example['intent_label']].append(i)

        print("Populated example cache.")

    # One example for each label
    example_inds = []
    for l in set(labels.tolist()):
        if l == -1:
            continue

        ind = random.choice(cache[l])
        retries = 0
        while ind in inds.tolist() or type(ind) is not int:
            ind = random.choice(cache[l])
            retries += 1
            if retries > len(dataset):
                break

        example_inds.append(ind)

    # Sample randomly until we hit batch size
    while len(example_inds) < min(len(dataset), num):
        ind = random.randint(0, len(dataset) - 1)
        if ind not in example_inds and ind not in inds.tolist():
            example_inds.append(ind)

    # Create examples
    example_data = [dataset[i] for i in example_inds]
    examples = {}
    for key in ['input_ids', 'attention_mask', 'token_type_ids']:
        examples[key] = torch.stack([torch.LongTensor(e[key]) for e in example_data], dim=0).cuda()

    examples['intent_label'] = torch.LongTensor([e['intent_label'] for e in example_data]).cuda()

    return examples


def evaluate(model: torch.nn.Module,
             eval_dataloader: DataLoader,
             ex_dataloader: DataLoader,
             tokenizer: Any,
             task: str = "intent",
             example: bool = False,
             device: int = 0,
             args: Any = None) -> Tuple[float, float, float]:
    model.eval()

    output = []
    labels = []
    if example:
        assert task == "intent", "Example-Driven may only be used for intent prediction"

        with torch.no_grad():
            for batch in tqdm(ex_dataloader, desc="Building train memory."):
                # Move to GPU
                if torch.cuda.is_available():
                    for key, val in batch.items():
                        if type(batch[key]) is list:
                            continue

                        batch[key] = batch[key].to(device)

                pooled_output = model.encode(batch["input_ids"], batch["attention_mask"], batch["token_type_ids"])

                output.append(pooled_output.cpu())
                labels += batch["intent_label"].tolist()

            mem = torch.cat(output, dim=0).cuda()
            print("Memory size:", mem.size())

    pred = []
    true = []
    input = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            # Move to GPU
            if torch.cuda.is_available():
                for key, val in batch.items():
                    if type(batch[key]) is list:
                        continue

                    batch[key] = batch[key].to(device)
            if args.generation:
                if task == "intent" :
                    if args.petgen:
                        result = model.generate(input_ids=batch["input_ids"],eos_token_id=tokenizer.eos_token_id,
                        max_length=20, do_sample=True) # add do_sample to increase diversity
                    else:
                        result = model.generate(input_ids=batch["input_ids"],eos_token_id=tokenizer.eos_token_id,max_length=20)
                    # attention_mask is not needed
                    results = result.cpu().tolist()
                    #if args.petgen:
                    input_batch = batch["origin_input"].cpu().tolist() # input_ids can also be added
                    for i in range(len(results)):
                        pred.append(tokenizer.decode(results[i], skip_special_tokens=True))  # [0]
                        true.append(tokenizer.decode(batch["intent_ids"].cpu().tolist()[i], skip_special_tokens=True))
                        # if args.petgen:
                        input.append(tokenizer.decode(input_batch[i], skip_special_tokens=True))
                if task == "slot":
                    #if args.petgen:
                    #    result = model.generate(input_ids=batch["input_ids"],eos_token_id=tokenizer.eos_token_id,
                    #    max_length=20, do_sample=True) # 
                    # do not add do_sample to increase accuracy
                    result = model.generate(input_ids=batch["input_ids"],attention_mask = batch["attention_mask"],eos_token_id=tokenizer.eos_token_id,max_length=40)
                    # batch_size 1
                    results = result.cpu().tolist()
                    if args.petgen:
                        input_batch = batch["origin_input"].cpu().tolist()
                    for i in range(len(results)):
                        pred.append(tokenizer.decode(results[i], skip_special_tokens=True))  # [0]
                        true.append(tokenizer.decode(batch["slot_ids"].cpu().tolist()[i], skip_special_tokens=True))
                        if args.petgen:
                            input.append(tokenizer.decode(input_batch[i], skip_special_tokens=True))
                if task == "top":
                    if args.prompt == ['file1']:
                        intent_result = model.generate(input_ids=batch["input_intent"], attention_mask=batch["attention_intent"], eos_token_id=tokenizer.eos_token_id, max_length=20)
                        slot_result = model.generate(input_ids=batch["input_slot"], attention_mask=batch["attention_slot"], eos_token_id=tokenizer.eos_token_id, max_length=60)
                    else:
                        additional_attention = torch.ones(batch["input_ids"].shape[0],6).cuda() # batch["intent_prompt"].shape[1]
                        intent_result = model.generate(input_ids=torch.cat((batch["input_ids"],batch["intent_prompt"]),dim = 1),attention_mask=torch.cat((batch["attention_mask"],additional_attention),dim = 1),eos_token_id=tokenizer.eos_token_id,max_length=20)
                        slot_result = model.generate(input_ids=torch.cat((batch["input_ids"],batch["slot_prompt"]),dim = 1),attention_mask=torch.cat((batch["attention_mask"],additional_attention),dim = 1),eos_token_id=tokenizer.eos_token_id,max_length=80) # 100
                    # batch_size 1
                    intent_results = intent_result.cpu().tolist()
                    slot_results = slot_result.cpu().tolist()
                    if args.petgen:
                        input_batch = batch["origin_input"].cpu().tolist()
                    for i in range(len(intent_results)):
                        pred.append((tokenizer.decode(intent_results[i], skip_special_tokens=True),tokenizer.decode(slot_results[i], skip_special_tokens=True)))  # [0]
                        true.append((tokenizer.decode(batch["intent_ids"].cpu().tolist()[i], skip_special_tokens=True),tokenizer.decode(batch["slot_ids"].cpu().tolist()[i], skip_special_tokens=True)))
                        if args.petgen:
                            input.append(tokenizer.decode(input_batch[i], skip_special_tokens=True))
            else:
                if task == "intent":
                    if not example:
                        # Forward prop
                        intent_logits, intent_loss = model(input_ids=batch["input_ids"],
                                                        attention_mask=batch["attention_mask"],
                                                        token_type_ids=batch["token_type_ids"],
                                                        intent_label=batch["intent_label"])

                        # Argmax to get predictions
                        intent_preds = torch.argmax(intent_logits, dim=1).cpu().tolist()

                        pred += intent_preds
                        true += batch["intent_label"].cpu().tolist()
                    else:
                        # Encode input
                        pooled_output = model.encode(batch["input_ids"], batch["attention_mask"], batch["token_type_ids"])

                        # Probability distribution over examples
                        probs = torch.softmax(pooled_output.mm(mem.t())[0], dim=-1)

                        # Copy mechanism over training set
                        intent_probs = torch.zeros(len(ex_dataloader.dataset.intent_idx_to_label)).cuda().scatter_add(0,
                                                                                                                    torch.LongTensor(
                                                                                                                        labels).cuda(),
                                                                                                                    probs)

                        pred.append(intent_probs.argmax(dim=-1).item())
                        true += batch["intent_label"].cpu().tolist()
                elif task == "slot":
                    # Forward prop
                    slot_logits, slot_loss = model(input_ids=batch["input_ids"],
                                                attention_mask=batch["attention_mask"],
                                                token_type_ids=batch["token_type_ids"],
                                                slot_labels=batch["slot_labels"])

                    # Argmax to get predictions
                    slot_preds = torch.argmax(slot_logits, dim=2).detach().cpu().numpy()

                    # Generate words, true slots and pred slots
                    words = [tokenizer.decode([e]) for e in batch["input_ids"][0].tolist()]
                    actual_gold_slots = batch["slot_labels"].cpu().numpy().squeeze().tolist()
                    true_slots = [eval_dataloader.dataset.slot_idx_to_label[s] for s in actual_gold_slots]
                    actual_predicted_slots = slot_preds.squeeze().tolist()
                    pred_slots = [eval_dataloader.dataset.slot_idx_to_label[s] for s in actual_predicted_slots]

                    # Find the last turn and only include that. Irrelevant for restaurant8k/dstc8-sgd.
                    if '>' in words:
                        ind = words[::-1].index('>')
                        words = words[-ind:]
                        true_slots = true_slots[-ind:]
                        pred_slots = pred_slots[-ind:]

                    # Filter out words that are padding
                    filt_words = [w for w in words if w not in ['', 'user']]
                    true_slots = [s for w, s in zip(words, true_slots) if w not in ['', 'user']]
                    pred_slots = [s for w, s in zip(words, pred_slots) if w not in ['', 'user']]

                    # Convert to slot labels
                    pred.append(pred_slots)
                    true.append(true_slots)

                    assert len(pred_slots) == len(true_slots)
                    assert len(pred_slots) == len(filt_words)
                elif task == "top":
                    intent_logits, slot_logits, _ = model(input_ids=batch["input_ids"],
                                                        attention_mask=batch["attention_mask"],
                                                        token_type_ids=batch["token_type_ids"])

                    # Argmax to get intent predictions
                    intent_preds = torch.argmax(intent_logits, dim=1).cpu().tolist()

                    # Argmax to get slot predictions
                    slot_preds = torch.argmax(slot_logits, dim=2).detach().cpu().numpy()
                    actual_predicted_slots = slot_preds.squeeze().tolist()
                    intent_true = batch["intent_label"].cpu().tolist()
                    actual_gold_slots = batch["slot_labels"].cpu().numpy().squeeze().tolist()

                    # Only unmasked
                    pad_ind = len(batch["attention_mask"].tolist()[0])#batch["attention_mask"].tolist()[0].index(0)
                    actual_gold_slots = actual_gold_slots[1:pad_ind - 1]
                    actual_predicted_slots = actual_predicted_slots[1:pad_ind - 1]

                    # Add to lists
                    pred.append((intent_preds if type(intent_preds) is int else intent_preds[0], actual_predicted_slots))
                    true.append((intent_true[0], actual_gold_slots))
    
    def gen_extract(slot_labels):
        """
        Convert from IBO slot labels to spans.
        """
        slots = []
        if 'restaurant8k' in args.train_data_path:
            all_slots = ['date', 'last_name', 'time', 'people', 'first_name']  #can be get from vocab.txt
        elif 'dstc8' in args.train_data_path:
            all_slots = ['leaving_date', 'visit_date', 'date', 'subcategory', 'to_location', 'area', 'dropoff_date', 'pickup_city', 'city_of_event', 'pickup_time', 'pickup_date', 'from_location', 'event_name']
        #elif 'top' in args.train_data_path:
        #    all_slots = ['leaving_date', 'visit_date', 'date', 'subcategory', 'to_location', 'area', 'dropoff_date', 'pickup_city', 'city_of_event', 'pickup_time', 'pickup_date', 'from_location', 'event_name']
        slot_labels_list = slot_labels.split()
        for slot in all_slots:
            if slot in slot_labels_list:
                start = slot_labels_list.index(slot)
                end = 0
                for part in slot_labels_list[ (start+1) :]:
                    if (part in all_slots) and (part!=slot):
                        break
                    end = end + 1
                    
                label = slot_labels_list[start+1 :start+end+1]
                slots.append("{}:{}".format(slot, label))

        return slots

    def _extract(slot_labels):
        """
        Convert from IBO slot labels to spans.
        """
        slots = []
        cur_key = None
        start_ind = -1
        for i, s in enumerate(slot_labels):
            if s == "O" or s == "[PAD]":
                # Add on-going slot if there is one
                if cur_key is not None:
                    slots.append("{}:{}-{}".format(cur_key, start_ind, i))

                cur_key = None
                continue

            token_type, slot_key = s.split("-", 1)
            if token_type == "B":
                # If there is an on-going slot right now, add it
                if cur_key is not None:
                    slots.append("{}:{}-{}".format(cur_key, start_ind, i))

                cur_key = slot_key
                start_ind = i
            elif token_type == "I":
                # If the slot key doesn't match the currently active, this is invalid.
                # Treat this as an O.
                if slot_key != cur_key:
                    if cur_key is not None:
                        slots.append("{}:{}-{}".format(cur_key, start_ind, i))

                    cur_key = None
                    continue

        # After the loop, add any oongoing slots
        if cur_key is not None:
            slots.append("{}:{}-{}".format(cur_key, start_ind, len(slot_labels)))

        return slots

    # Perform evaluation
    if task == "intent":
        result_dict = {}
        if args.dump_outputs:
            #pred_labels = [eval_dataloader.dataset.intent_idx_to_label.get(p) for p in pred]
            if args.petgen:
                result = [(input[i], pred[i]) for i in range(len(pred))]
                json.dump(result, open(args.output_dir + "/outputs.json", "w"))
            else:
                json.dump(pred, open(args.output_dir + "/outputs.json", "w")) # w+
        for input_num in range(len(input)):
            input_case = input[input_num]
            predict = pred[input_num]
            ground = true[input_num]
            if input_case not in result_dict:
                result_dict[input_case] = {'pred':{},'true':{}}
            if predict not in result_dict[input_case]['pred']:
                result_dict[input_case]['pred'][predict] = 0
            result_dict[input_case]['pred'][predict] += 1
            if ground not in result_dict[input_case]['true']:
                result_dict[input_case]['true'][ground] = 0
            result_dict[input_case]['true'][ground] += 1
        voting_acc = 0
        def get_voting(input):
            max = 0
            result =''
            for i, num in input.items():
                if num>max:
                    max = num
                    result = i
            return result
        for _, result_input in result_dict.items():
            predicted = get_voting(result_input['pred'])
            ground_t = get_voting(result_input['true'])
            voting_acc += predicted==ground_t
        voting_acc =voting_acc/len(result_dict)
        print(f"voting acc: {voting_acc}") 
        # if voting acc are used to select the model, please return voting acc and print acc
        return sum(p == t for p, t in zip(pred, true)) / len(pred)
    elif task == "slot":
        if args.generation:
            pred_slots = [gen_extract(e) for e in pred]
            true_slots = [gen_extract(e) for e in true]
            fp = 0
            fn = 0
            count = 0
            acc = 0
            result =[]
            """ the standard evaluation will yield higher score
            slot_types = set([slot.split(":")[0] for row in true_slots for slot in row])
            slot_type_f1_scores = []

            for slot_type in slot_types:
                predictions_for_slot = [
                    [p for p in prediction if slot_type in p] for prediction in pred_slots
                ]
                labels_for_slot = [
                    [l for l in label if slot_type in l] for label in true_slots
                ]

                proposal_made = [len(p) > 0 for p in predictions_for_slot]
                has_label = [len(l) > 0 for l in labels_for_slot]
                prediction_correct = [
                    prediction == label for prediction, label in zip(predictions_for_slot, labels_for_slot)
                ]
                true_positives = sum([
                    int(proposed and correct)
                    for proposed, correct in zip(proposal_made, prediction_correct)
                ])
                num_predicted = sum([int(proposed) for proposed in proposal_made])
                num_to_recall = sum([int(hl) for hl in has_label])

                precision = true_positives / (1e-5 + num_predicted)
                recall = true_positives / (1e-5 + num_to_recall)

                f1_score = 2 * precision * recall / (1e-5 + precision + recall)
                slot_type_f1_scores.append(f1_score)
            print(np.mean(slot_type_f1_scores))
            return np.mean(slot_type_f1_scores)
            """
            for num in range(len(pred_slots)):
                result.append({'true':true[num],'pred':pred[num]})
                pred_item = pred_slots[num]
                true_item = true_slots[num]
                #if true_item!=[] or pred_item!=[]:
                if true_item!=[] or pred_item!=[]:
                    count = count + 1
                    if set(pred[num].split()) == set(true[num].split()):
                    # if pred[num] == true[num]: # try a simple evaluation
                        acc = acc + 1 # acc is a strict evaluation, means tp
                    p_flag = 1
                    r_flag = 1
                    for true_s in true_item:
                        if true_s not in pred_item:
                            r_flag = 0
                            break
                    for pred_s in pred_item:
                        if pred_s not in true_item:
                            p_flag = 0
                            break
                    fp = fp + 1 - p_flag 
                    fn = fn + 1 - r_flag
            precision = acc/(acc+ fp + 0.0000001)
            recall = acc/(acc+ fn + 0.0000001)
            f1 = 2*precision*recall/(precision + recall+ 0.0000001)
            acc = acc / count
            print('precision:{},recall:{},f1:{},acc:{}'.format(precision,recall,f1,acc))#,acc
            if args.petgen:
                result = [(input[i], pred[i]) for i in range(len(pred))]
                json.dump(result, open(args.output_dir + "/outputs.json", "w"))
            else:
                json.dump(result, open(args.output_dir + "outputs.json", "w"),indent=2)
            #"""

            
        else:
            pred_slots = [_extract(e) for e in pred]
            true_slots = [_extract(e) for e in true]

        if args.dump_outputs:
            json.dump(pred_slots, open(args.output_dir + "outputs.json", "w+"))

        slot_types = set([slot.split(":")[0] for row in true_slots for slot in row])
        slot_type_f1_scores = []

        for slot_type in slot_types:
            predictions_for_slot = [
                [p for p in prediction if slot_type in p] for prediction in pred_slots
            ]
            labels_for_slot = [
                [l for l in label if slot_type in l] for label in true_slots
            ]

            proposal_made = [len(p) > 0 for p in predictions_for_slot]
            has_label = [len(l) > 0 for l in labels_for_slot]
            prediction_correct = [
                prediction == label for prediction, label in zip(predictions_for_slot, labels_for_slot)
            ]
            true_positives = sum([
                int(proposed and correct)
                for proposed, correct in zip(proposal_made, prediction_correct)
            ])
            num_predicted = sum([int(proposed) for proposed in proposal_made])
            num_to_recall = sum([int(hl) for hl in has_label])

            precision = true_positives / (1e-5 + num_predicted)
            recall = true_positives / (1e-5 + num_to_recall)

            f1_score = 2 * precision * recall / (1e-5 + precision + recall)
            slot_type_f1_scores.append(f1_score)
        if args.generation:
            print(np.mean(slot_type_f1_scores))
            return f1
        else:
            return np.mean(slot_type_f1_scores)
    elif task == "top":
        if args.petgen:
            result_intent = [(input[i], pred[i][0]) for i in range(len(pred))]
            json.dump(result_intent, open(args.output_dir + "/outputs_intent.json", "w"))
            result_slot = [(input[i], pred[i][1]) for i in range(len(pred))]
            json.dump(result_slot, open(args.output_dir + "/outputs_slot.json", "w"))
        else:
            if args.dump_outputs:
                pred_labels = [(eval_dataloader.dataset.intent_idx_to_label[intent],
                                [eval_dataloader.dataset.slot_idx_to_label[e] for e in slots]) for intent, slots in pred]
                json.dump(pred_labels, open(args.output_dir + "outputs.json", "w+"))

        return sum(p == t for p, t in zip(pred, true)) / len(pred)


def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    # special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(labels == 0, dtype=torch.bool), value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.token_to_id("[MASK]")

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(tokenizer.get_vocab_size(), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random].cuda()

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def train(args, rep):
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Rename output dir based on arguments
    if args.output_dir == "":
        cwd = os.getcwd()
        base = args.model_name_or_path.split("/")[-1]
        model_type = "_example" if args.example else "_linear"
        data_path = '_' + '_'.join(args.train_data_path.split("/")[-2:]).replace(".csv", "")
        mlm_on = "_mlmtrain" if args.mlm_data_path == "" or args.mlm_data_path == args.train_data_path else "_mlmfull"
        mlm_pre = "_mlmpre" if args.mlm_pre else ""
        mlm_dur = "_mlmdur" if args.mlm_during else ""
        observer = "_observer" if args.use_observers else ""
        name = base + model_type + data_path + mlm_on + mlm_pre + mlm_dur + observer + "_v{}".format(rep)
        args.output_dir = os.path.join(cwd, "checkpoints", name)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    elif args.num_epochs == 0:
        # This means we're evaluating. Don't create the directory.
        pass
    #else:
    #    raise Exception("Directory {} already exists".format(args.output_dir))

    # Dump arguments to the checkpoint directory, to ensure reproducability.
    if args.num_epochs > 0:
        json.dump(args.__dict__, open(os.path.join(args.output_dir, 'args.json'), "w+"))
        torch.save(args, os.path.join(args.output_dir, "run_args"))

    # Configure tensorboard writer
    tb_writer = SummaryWriter(log_dir=args.output_dir)

    # Configure tokenizer
    token_vocab_name = 't5'
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)#additional_special_tokens = special_tokens
    #tokenizer.enable_padding(length=args.max_seq_length)

    #if args.num_epochs > 0:
    #    tokenizer.save(args.output_dir)

        # Data readers
    if args.task == "intent":
        dataset_initializer = IntentDataset
    elif args.task == "slot":
        dataset_initializer = SlotDataset
    elif args.task == "top":
        dataset_initializer = TOPDataset
    else:
        raise ValueError("Not a valid task type: {}".format(args.task))

    train_dataset = dataset_initializer(args.train_data_path,
                                        tokenizer,
                                        args.max_seq_length,
                                        token_vocab_name,
                                        prompt=args.prompt,
                                        pet=(rep if args.prepet else None),
                                        pet_final=args.pet_final)

    val_dataset = dataset_initializer(args.val_data_path,
                                      tokenizer,
                                      512,
                                      token_vocab_name,
                                      prompt=args.prompt,
                                      pet=(rep if args.prepet else None)) if args.val_data_path else None

    test_dataset = dataset_initializer((args.pet_data_path if args.petgen else args.test_data_path),
                                       tokenizer,
                                       512,
                                       token_vocab_name,
                                       prompt=args.prompt,
                                       pet=(rep if (args.prepet or args.petgen) else None))
    if args.mlm_data_path != '':
        mlm_dataset = dataset_initializer(args.mlm_data_path,
                                          tokenizer,
                                          args.max_seq_length,
                                          token_vocab_name,
                                          prompt=args.prompt)
    else:
        mlm_dataset = train_dataset
    # Data loaders

    def collate_fn(self,tokenizer = tokenizer):
        # suit for list of dicts
        pad_id = tokenizer.pad_token_id
        pad_result = {}
        """
        utt = []
        label = []
        for items in self:
            utt.append(items['input_utt'])
            label.append(items['intent_utt'])
        pad_result['input_ids'] = tokenizer(utt,pad_id = pad_id)
        pad_result['ouput_ids'] = tokenizer(label)
        pad_batch=np.ones(len(self))
        for idx, s in enumerate(self):
            pad_batch[idx] = s['intent_label']
        pad_result['intent_label'] = torch.from_numpy(pad_batch).long()
        return pad_result

        """
        for key in self[0]:# same pad_id for all values
            
            if  not isinstance(self[0][key],int): # can be modified
                max_len = max(len(input[key]) for input in self)
                #batch_size = len(samples)
                #if key == 'intent_label':
                #    pad_batch=np.ones((len(self), max_len))*(-100)
                #else:
                pad_batch=np.ones((len(self), max_len))*pad_id  #-100
                for idx, s in enumerate(self):
                    #trunc = s[-max_len:]
                    pad_batch[idx, :len(s[key])] = s[key]
                pad_result[key] = torch.from_numpy(pad_batch).long()
            
            else: # labels
                pad_batch=np.ones(len(self))
                for idx, s in enumerate(self):
                    pad_batch[idx] = s[key]
                pad_result[key] = torch.from_numpy(pad_batch).long()
        return pad_result  

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.train_batch_size,
                                  collate_fn=collate_fn,
                                  shuffle=True,
                                  pin_memory=True)

    mlm_dataloader = DataLoader(dataset=mlm_dataset,
                                batch_size=args.train_batch_size,
                                #collate_fn=collate_fn,
                                shuffle=True,
                                pin_memory=True)

    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=args.eval_batch_size,  #  1
                                collate_fn=collate_fn,
                                pin_memory=True) if val_dataset else None

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn,
                                 #shuffle=True,
                                 pin_memory=True)

    # Load model
    
    if args.task == "intent":
        
        if args.example:
            model = ExampleIntentBertModel(args.model_name_or_path,
                                           dropout=args.dropout,
                                           num_intent_labels=len(train_dataset.intent_label_to_idx),
                                           use_observers=args.use_observers)
        else:
            model = IntentT5Model(args.model_name_or_path,
                                    dropout=args.dropout,
                                    num_intent_labels=len(train_dataset.intent_label_to_idx),
                                    use_observers=args.use_observers)
        
    elif args.task == "slot":
        model = SlotT5Model(args.model_name_or_path,
                              dropout=args.dropout,
                              num_slot_labels=len(train_dataset.slot_label_to_idx))
    elif args.task == "top":
        model = JointSlotIntentT5Model(args.model_name_or_path,
                                         dropout=args.dropout,
                                         num_intent_labels=len(train_dataset.intent_label_to_idx),
                                         num_slot_labels=len(train_dataset.slot_label_to_idx))
    else:
        raise ValueError("Cannot instantiate model for task: {}".format(args.task))
    if args.generation:
        model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    #model = T5Pretrain(args.model_name_or_path)
    if torch.cuda.is_available():
        model.to(args.device)

    # Initialize MLM model
    if args.mlm_pre or args.mlm_during:
        pre_model = T5Pretrain(args.model_name_or_path)
        mlm_optimizer = AdamW(pre_model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
        if torch.cuda.is_available():
            pre_model.to(args.device)

    # MLM Pre-train
    if args.mlm_pre and args.num_epochs > 0:
        # Maintain most recent score per label.
        for epoch in trange(3, desc="Pre-train Epochs"):
            pre_model.train()
            epoch_loss = 0
            num_batches = 0
            for batch in tqdm(mlm_dataloader):
                num_batches += 1

                # Train model
                if "input_ids" in batch:
                    inputs, labels = mask_tokens(batch["input_ids"].cuda(), tokenizer)
                else:
                    inputs, labels = mask_tokens(batch["ctx_input_ids"].cuda(), tokenizer)

                loss = pre_model(inputs, labels)
                if args.grad_accum > 1:
                    loss = loss / args.grad_accum
                loss.backward()
                epoch_loss += loss.item()

                if args.grad_accum <= 1 or num_batches % args.grad_accum == 0:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(pre_model.parameters(), args.max_grad_norm)

                    mlm_optimizer.step()
                    pre_model.zero_grad()

            LOGGER.info("Epoch loss: {}".format(epoch_loss / num_batches))

        # Transfer BERT weights
        #model.bert_model = pre_model.bert_model.bert

    # Train
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    global_step = 0
    metrics_to_log = {}
    best_score = -1
    patience = 0
    for epoch in trange(args.num_epochs, desc="Epoch"):
        model.train()
        epoch_loss = 0
        num_batches = 0

        for batch in tqdm(train_dataloader):
            num_batches += 1
            global_step += 1

            # Transfer to gpu
            if torch.cuda.is_available():
                for key, val in batch.items():
                    if type(batch[key]) is list:
                        continue

                    batch[key] = batch[key].to(args.device)
            if args.generation:
                if args.task == "intent":
                    batch["intent_ids"][batch["intent_ids"] == tokenizer.pad_token_id] = -100 # ignore pad_id in generation
                    loss = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],labels=batch["intent_ids"], return_dict=False)[0]
                    loss.backward()
                    epoch_loss += loss.item()
                if args.task == "slot":
                    batch["slot_ids"][batch["slot_ids"] == tokenizer.pad_token_id] = -100
                    loss = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],labels=batch["slot_ids"], return_dict=False)[0]
                    loss.backward()
                    epoch_loss += loss.item()
                if args.task == "top":
                    #batch["intent_prompt"]
                    #batch["slot_prompt"]
                    if 'intent_ids' in batch: # dealing with pet input
                        batch["intent_ids"][batch["intent_ids"] == tokenizer.pad_token_id] = -100
                    if 'slot_ids' in batch:
                        batch["slot_ids"][batch["slot_ids"] == tokenizer.pad_token_id] = -100
                    if args.prompt == ['file1']:
                        if 'intent_ids' in batch:
                            loss_intent = model(input_ids=batch["input_intent"],
                            attention_mask=batch["attention_intent"], labels=batch["intent_ids"], return_dict=False)[0]
                        else:
                            loss_intent = 0.0
                        if 'slot_ids' in batch:
                            loss_slot = model(input_ids=batch["input_slot"],
                            attention_mask=batch["attention_slot"], labels=batch["slot_ids"], return_dict=False)[0]
                        else:
                            loss_slot = 0.0  
                    else:
                        additional_attention = torch.ones(batch["input_ids"].shape[0],6).cuda() #batch_size, 6 need to be changed to batch["intent_prompt"].shape[1] and batch["slot_prompt"].shape[1]
                        loss_intent = model(input_ids=torch.cat((batch["input_ids"],batch["intent_prompt"]),dim = 1),
                        attention_mask=torch.cat((batch["attention_mask"],additional_attention),dim = 1), labels=batch["intent_ids"], return_dict=False)[0] 
                        loss_slot = model(input_ids=torch.cat((batch["input_ids"],batch["slot_prompt"]),dim = 1),
                        attention_mask=torch.cat((batch["attention_mask"],additional_attention),dim = 1), labels=batch["slot_ids"], return_dict=False)[0] #attention_mask=batch["attention_mask"]
                    loss = loss_intent + loss_slot
                    loss.backward()
                    epoch_loss += loss.item()
            else:
            # Train model
                if args.task == "intent":
                    if args.example:
                        examples = retrieve_examples(train_dataset, batch["intent_label"], batch["ind"], task="intent")

                        _, intent_loss = model(input_ids=batch["input_ids"],
                                            attention_mask=batch["attention_mask"],
                                            token_type_ids=batch["token_type_ids"],
                                            intent_label=batch["intent_label"],
                                            example_input=examples["input_ids"],
                                            example_mask=examples["attention_mask"],
                                            example_token_types=examples["token_type_ids"],
                                            example_intents=examples["intent_label"])
                    else:
                        _, intent_loss = model(input_ids=batch["input_ids"],
                                            attention_mask=batch["attention_mask"],
                                            token_type_ids=batch["token_type_ids"],
                                            intent_label=batch["intent_label"])
                    if args.grad_accum > 1:
                        intent_loss = intent_loss / args.grad_accum
                    intent_loss.backward()
                    epoch_loss += intent_loss.item()
                elif args.task == "slot":
                    _, slot_loss = model(input_ids=batch["input_ids"],
                                        attention_mask=batch["attention_mask"],
                                        token_type_ids=batch["token_type_ids"],
                                        slot_labels=batch["slot_labels"])

                    if args.grad_accum > 1:
                        slot_loss = slot_loss / args.grad_accum
                    slot_loss.backward()
                    epoch_loss += slot_loss.item()
                elif args.task == "top":
                    _, _, loss = model(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"],
                                    token_type_ids=batch["token_type_ids"],
                                    intent_label=batch["intent_label"],
                                    slot_labels=batch["slot_labels"])

                    if args.grad_accum > 1:
                        loss = loss / args.grad_accum
                    loss.backward()
                    epoch_loss += loss.item()

            if args.grad_accum <= 1 or num_batches % args.grad_accum == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                model.zero_grad()

        LOGGER.info("Epoch loss: {}".format(epoch_loss / num_batches))

        # Evaluate and save checkpoint
        score = evaluate(model, test_dataloader, train_dataloader, tokenizer, task=args.task, example=args.example,
                         device=args.device, args=args) # val_dataloader
        metrics_to_log["eval_score"] = score
        LOGGER.info("Task: {}, score: {}---".format(args.task,
                                                    score))

        if score < best_score:
            patience += 1
        else:
            patience = 0

        if score > best_score:
            LOGGER.info("New best results found for {}! Score: {}".format(args.task,
                                                                          score))
            torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
            torch.save(optimizer.state_dict(), os.path.join(args.output_dir, "optimizer.pt"))
            best_score = score

        for name, val in metrics_to_log.items():
            tb_writer.add_scalar(name, val, global_step)

        if patience >= args.patience:
            LOGGER.info("Stopping early due to patience")
            break

        # Run MLM during training
        if args.mlm_during:
            pre_model.train()
            epoch_loss = 0
            num_batches = 0
            for batch in tqdm(mlm_dataloader):
                num_batches += 1

                # Train model
                if "input_ids" in batch:
                    inputs, labels = mask_tokens(batch["input_ids"].cuda(), tokenizer)
                else:
                    inputs, labels = mask_tokens(batch["ctx_input_ids"].cuda(), tokenizer)

                loss = pre_model(inputs, labels)

                if args.grad_accum > 1:
                    loss = loss / args.grad_accum

                loss.backward()
                epoch_loss += loss.item()

                if args.grad_accum <= 1 or num_batches % args.grad_accum == 0:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(pre_model.parameters(), args.max_grad_norm)

                    mlm_optimizer.step()
                    pre_model.zero_grad()

            LOGGER.info("MLMloss: {}".format(epoch_loss / num_batches))

    # Evaluate on test set
    LOGGER.info("Loading up best model for test evaluation...")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "model.pt")))
    score = evaluate(model, test_dataloader, train_dataloader, tokenizer, task=args.task, example=args.example,
                     device=args.device, args=args)
    print("Best result for {}: Score: {}".format(args.task, score))
    tb_writer.add_scalar("final_test_score", score, global_step)
    tb_writer.close()
    return score


if __name__ == "__main__":
    # PETNUM = 5
    PETNUM = 25 # for pet ablation
    args = read_args()
    print(args)

    scores = []
    seeds = [34]
    #34
    #seeds = [33, 42, 19, 55, 34, 63]
    if args.prepet or args.petgen:
        for i in range(PETNUM):
            checkpoint_base = 'checkpoint4/epoch' if (('dstc8' not in args.train_data_path) and ('top' not in args.train_data_path)) else 'checkpoint8/epoch' 
            # avoid data leakage in dstc8, use multi prompts in top
            #num = 15 + i # 10 + i
            # 10 + i
            num=10
            args.seed = seeds[0]
            args.model_name_or_path = checkpoint_base + str(num)
            args.output_dir = args.output_dir + str(i)
            score=train(args, i)
            print(f"Score for pet {i}: {score}")
    else:
        for i in range(args.repeat):
            #if args.num_epochs > 0:
            #    args.output_dir = ""

            args.seed = seeds[i] if i < len(seeds) else random.randint(1, 999)
            scores.append(train(args, i))

            print("Average score so far:", np.mean(scores))

        print(scores)
        print(np.mean(scores), max(scores), min(scores))
