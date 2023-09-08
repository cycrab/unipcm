import csv
import logging
import json
import numpy as np
import os
import pickle
import random

from collections import defaultdict
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Dict

from constants import SPECIAL_TOKENS

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)


class IntentDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 tokenizer: BertWordPieceTokenizer,
                 max_seq_length: int,
                 vocab_file_name: str,
                 prompt= [],
                 pet=None,
                 pet_final=''):
        # For caching
        data_dirname = os.path.dirname(os.path.abspath(data_path))
        split = os.path.basename(os.path.abspath(data_path))

        # Intent categories
        intent_vocab_path = os.path.join(data_dirname, "categories.json")
        if os.path.exists(intent_vocab_path):
            intent_names = json.load(open(intent_vocab_path))
            self.intent_label_to_idx = dict((label, idx) for idx, label in enumerate(intent_names))
            self.intent_idx_to_label = {idx: label for label, idx in self.intent_label_to_idx.items()}
        else: # hwu
            data = json.load(open(data_path))
            utt = []
            intent_text = []
            intent_label = []
            self.intent_label_to_idx = {}
            self.intent_idx_to_label = {}
            all_domains = []
            for id, dial in data.items():
                utt.append(dial['turns'][0]['text'])
                turn_domain = list(dial['turns'][0]['label']['DEFAULT_DOMAIN'].keys())[0]
                turn_intent = dial['turns'][0]['extra_info']['intent_label']
                if turn_domain not in self.intent_label_to_idx:
                    self.intent_label_to_idx[turn_domain] = turn_intent
                    self.intent_idx_to_label[turn_intent] = turn_domain
                intent_text.append(turn_domain)
                intent_label.append(turn_intent)
                 
        # Process data
        self.tokenizer = tokenizer
        cached_path = os.path.join(data_dirname, "{}_{}_intent".format(split, vocab_file_name) + '_'.join(prompt) + "_cached")
        if pet!=None:
            cached_path = cached_path + str(pet)
        if pet_final!='':
            cached_path = cached_path + 'pet'
        if not os.path.exists(cached_path):
            self.examples = []
            reader = csv.reader(open(data_path))
            if 'hwu' in data_path:
                reader = zip(utt,intent_text)
            next(reader, None)
            out = []
            def wrap_encoded(encoded, intent, origin_input=None):
                new_intent = intent.replace('_',' ')
                intent_encoded = tokenizer.encode(new_intent)
                attention=len(encoded)*[1] # no_mask
                intent_attention=len(intent_encoded)*[1] 
                case = {
                    "input_ids": np.array(encoded)[-max_seq_length:],
                    "attention_mask": np.array(attention)[-max_seq_length:],
                    "token_type_ids": np.array(attention)[-max_seq_length:],
                    "intent_ids": np.array(intent_encoded)[-max_seq_length:],
                    "intent_mask": np.array(intent_attention)[-max_seq_length:],
                    "intent_label": self.intent_label_to_idx[intent] if intent in self.intent_label_to_idx else 0, # need modification with classification
                    "ind": len(self.examples),
                }
                if origin_input:
                    case['origin_input'] = np.array(origin_input)[-max_seq_length:]
                self.examples.append(case)
            if prompt!=['t0']:
                for utt, intent in tqdm(reader):
                    
                    if prompt == []:
                        encoded = tokenizer.encode(utt)
                    elif len(prompt)==1:
                        if prompt[0]!='file' and prompt[0]!='file1' :
                            encoded = tokenizer.encode(utt + ('. ' if ('.' not in utt and '?' not in utt) else ' ') + prompt[0])
                        else:
                            if prompt[0]=='file':
                                prompts = json.load(open('dialoglue/intent_prompts.json','r'))
                            elif prompt[0]=='file1':
                                if pet!=None:
                                    prompts = json.load(open('data/new_prompts/intent_pet.json','r'))[pet]
                                else:
                                    prompts = json.load(open('data/new_prompts/intentsimplified_selected_dialoglue.json','r')) # [:7]
                            if ('train' in split): #or ('val' in split): 
                                # train_pet in this mode now, it's logical but can be modified
                                for tmp_prompt in prompts:
                                    encoded = tokenizer.encode(tmp_prompt[0] + ' ' + utt + ('. ' if ('.' not in utt and '?' not in utt) else ' ') + tmp_prompt[1])
                                    if pet==None:
                                        wrap_encoded(encoded, intent)
                                    else:
                                        wrap_encoded(encoded, intent, tokenizer.encode(utt))
                            elif 'val' in split:
                                eval_num = 5 if (prompt[0]=='file1' and (pet==None)) else 3
                                tmp_prompts = random.sample(prompts,eval_num)
                                for i in range(eval_num):
                                    tmp_prompt = tmp_prompts[i]
                                    encoded = tokenizer.encode(tmp_prompt[0] + ' ' + utt + ('. ' if ('.' not in utt and '?' not in utt) else ' ') + tmp_prompt[1])
                                    wrap_encoded(encoded, intent)
                            else:
                                test_num = 5 if (prompt[0]=='file1' and (pet==None)) else 3
                                tmp_prompt = random.sample(prompts,test_num)[0]
                                encoded = tokenizer.encode(tmp_prompt[0] + ' ' + utt + ('. ' if ('.' not in utt and '?' not in utt) else ' ') + tmp_prompt[1])
                                wrap_encoded(encoded, intent)
                    elif len(prompt)==2:
                        encoded = tokenizer.encode(prompt[0] + ' ' + utt + ('. ' if ('.' not in utt and '?' not in utt) else ' ') + prompt[1])
                    if prompt!=['file'] and prompt!=['file1']:
                        wrap_encoded(encoded, intent)
                if pet_final!='':
                    PET_AUG_NUM = 10
                    prompts = json.load(open('data/new_prompts/intentsimplified_selected_dialoglue.json','r'))
                    aug_data = json.load(open(pet_final,'r'))
                    for aug in aug_data:
                        utt = aug[0]
                        intent = aug[1]
                        tmp_prompts = random.sample(prompts, PET_AUG_NUM)
                        for tmp_prompt in tmp_prompts:
                            encoded = tokenizer.encode(tmp_prompt[0] + ' ' + utt + ('. ' if ('.' not in utt and '?' not in utt) else ' ') + tmp_prompt[1])
                            wrap_encoded(encoded, intent)

                with open(cached_path, "wb") as f:
                    pickle.dump(self.examples, f)
            else:
                from promptsource.templates import DatasetTemplates
                from datasets import load_dataset
                template_input = {}
                if 'banking' in data_dirname:
                    dataset = load_dataset('banking77')

                    prompts = DatasetTemplates('banking77')
                    train = dataset['train']
                    test = dataset['test']
                    #train = dataset['train']
                    if 'train' in split or 'val' in split:
                        all_input = []
                        #if '10' in split or '5' in split:
                        for utt, intent in reader:
                            all_input.append(utt)
                        for num in range(len(train)):
                            if (train[num]['text'] in all_input) :# or (('10' not in split) and ('5' not in split))
                                for t in prompts.all_template_names:
                                    if t not in template_input:
                                        template_input[t] = []
                                    prompt = prompts[t]
                                    result = prompt.apply(train[num])
                                    wrap_encoded(tokenizer.encode(result[0]), result[1])

                    elif 'test' in split:
                        for num in range(len(test)):
                            results = []
                            for t in prompts.all_template_names:
                                if t not in template_input:
                                    template_input[t] = []
                                prompt = prompts[t]
                                results.append(prompt.apply(train[num]))
                            result = random.sample(results,1)[0]
                            wrap_encoded(tokenizer.encode(result[0]), result[1])

                with open(cached_path, "wb") as f:
                    pickle.dump(self.examples, f)

        else:
            LOGGER.info("Loading from cached path: {}".format(cached_path))
            with open(cached_path, "rb") as f:
                self.examples = pickle.load(f)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

class SlotDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 tokenizer: BertWordPieceTokenizer,
                 max_seq_length: int,
                 vocab_file_name: str,
                 prompt=[],
                 pet=None,
                 pet_final=''):
        # For caching
        data_dirname = os.path.dirname(os.path.abspath(data_path))
        split = os.path.basename(os.path.abspath(data_path))

        # Slot categories
        slot_vocab_path = os.path.join(os.path.dirname(data_path), "vocab.txt")
        slot_names = json.load(open(slot_vocab_path))
        slot_names.insert(0, "[PAD]")
        self.slot_label_to_idx = dict((label, idx) for idx, label in enumerate(slot_names))
        self.slot_idx_to_label = {idx: label for label, idx in self.slot_label_to_idx.items()}

        # Process data
        self.tokenizer = tokenizer
        cached_path = os.path.join(data_dirname, "{}_{}_slots".format(split, vocab_file_name) + '_'.join(prompt) + "_cached")
        if pet!=None:
            cached_path = cached_path + str(pet)
        if pet_final!='':
            cached_path = cached_path + 'pet'
        if not os.path.exists(cached_path):
            self.examples = []
            data = json.load(open(data_path))
            for example in tqdm(data):
                text, slots, slot_text = self.parse_example(example) 
                def wrap_encoded(encoded, slot_text, origin_input=None):
                    attention=len(encoded)*[1]
                    encoded_slot_text = tokenizer.encode(slot_text)
                    case = {
                    "input_ids": np.array(encoded)[-max_seq_length:],
                    "attention_mask": np.array(attention)[-max_seq_length:],
                    "slot_ids": np.array(encoded_slot_text)[-max_seq_length:],
                    "ind": len(self.examples),
                    }
                    if origin_input:
                        case['origin_input'] = np.array(origin_input)[-max_seq_length:]
                    self.examples.append(case)
                    return
                utt = text + ('. ' if (('.' not in text) and ('?' not in text) and ('!' not in text)) else ' ')
                if len(prompt) == 1:
                    if prompt == ['file']:
                        prompts = json.load(open('data/new_prompts/state_dialoglue.json','r'))
                    elif prompt == ['file1']:
                        if pet!=None:
                            prompts = json.load(open('data/new_prompts/state_pet.json','r'))[pet]
                        else:
                            prompts = json.load(open('data/new_prompts/statesimplified_selected_dialoglue.json','r'))
                    if ('train' in split) or ('val' in split):
                        if 'val' in split:
                            prompts1 = random.sample(prompts,5) if (prompt == ['file1'] and (pet==None)) else random.sample(prompts,3)
                        else:
                            prompts1 = prompts
                        for tmp_prompt in prompts1:
                            encoded = tokenizer.encode( tmp_prompt[0] +' ' + utt + tmp_prompt[1]) # tmp_prompt[0] +
                            wrap_encoded(encoded, slot_text, tokenizer.encode(utt))
                    else:
                        tmp_prompts = random.sample(prompts,5) if (prompt == ['file1'] and (pet==None)) else random.sample(prompts,3) 
                        for tmp_prompt in tmp_prompts:
                            encoded = tokenizer.encode(tmp_prompt[0] +' ' + utt + tmp_prompt[1])
                            if pet==None:
                                wrap_encoded(encoded, slot_text)
                            else:
                                wrap_encoded(encoded, slot_text, tokenizer.encode(utt))
                elif len(prompt) == 2:
                    encoded = tokenizer.encode(prompt[0] + ' ' + text + ('. ' if ('.' not in text and '?' not in text) else ' ') + prompt[1])
                    wrap_encoded(encoded, slot_text)
                else:
                    encoded = tokenizer.encode(text)
                    attention=len(encoded)*[1]
                    type_ids=(len(encoded)-1)*[1] + [0]

                    encoded_slot_text = tokenizer.encode(slot_text)

                    encoded_slot_labels = self.encode_token_labels([text], [slots],
                                                                len(encoded),
                                                                tokenizer,
                                                                self.slot_label_to_idx,
                                                                max_seq_length)
                    self.examples.append({
                        "input_ids": np.array(encoded)[-max_seq_length:],
                        "attention_mask": np.array(attention)[-max_seq_length:],
                        "token_type_ids": np.array(type_ids)[-max_seq_length:],
                        "slot_ids": np.array(encoded_slot_text)[-max_seq_length:],
                        "slot_labels": encoded_slot_labels[-max_seq_length:],
                        "ind": len(self.examples),
                    })
            if pet_final!='':
                PET_AUG_NUM = 10
                prompts = json.load(open('data/new_prompts/statesimplified_selected_dialoglue.json','r'))
                aug_data = json.load(open(pet_final,'r'))
                for aug in aug_data:
                    utt = aug[0]
                    slot_text = aug[1]
                    tmp_prompts = random.sample(prompts, PET_AUG_NUM)
                    for tmp_prompt in tmp_prompts:
                        encoded = tokenizer.encode(tmp_prompt[0] + ' ' + utt + tmp_prompt[1])
                        wrap_encoded(encoded, slot_text, tokenizer.encode(utt))
            with open(cached_path, "wb") as f:
                pickle.dump(self.examples, f)
        else:
            LOGGER.info("Loading from cached path: {}".format(cached_path))
            with open(cached_path, "rb") as f:
                self.examples = pickle.load(f)

    def encode_token_labels(self,
                            text_sequences,
                            slot_names,
                            encoded_length,
                            tokenizer,
                            slot_map,
                            max_length) -> np.array:
    
        def _get_word_tokens_len(word: str, tokenizer: BertWordPieceTokenizer) -> int:
            return sum(map(lambda token: 1 if token not in tokenizer.all_special_tokens else 0,
                           tokenizer.tokenize(word)))
    
        encoded = np.zeros(shape=(len(text_sequences), encoded_length), dtype=np.int32)
        for i, (text_sequence, word_labels) in enumerate(
                zip(text_sequences, slot_names)):
            encoded_labels = []
            for word, word_label in zip(text_sequence.split(), word_labels.split()):
                encoded_labels.append(slot_map[word_label])
                expand_label = word_label.replace("B-", "I-")
                if not expand_label in slot_map:
                    expand_label = word_label
                word_tokens_len = _get_word_tokens_len(word, tokenizer)
                if ('.' in word or '?' in word) and word_tokens_len>=2:
                    encoded_labels.extend([slot_map[expand_label]] * (word_tokens_len - 2)+[1]) #o
                else:
                    encoded_labels.extend([slot_map[expand_label]] * (word_tokens_len - 1))
            encoded[i, 1:len(encoded_labels) + 1] = encoded_labels
        return encoded.squeeze()

    def parse_example(self, example):
        text = example['userInput']['text']
        slot_texts = ''
        # Create slots dictionary
        word_to_slot = {}
        for label in example.get('labels', []):
            slot = label['slot']
            start = label['valueSpan'].get('startIndex', 0)
            end = label['valueSpan'].get('endIndex', -1)
            if slot_texts == '':
                slot_texts = slot_texts + slot
            else:
                slot_texts = slot_texts + ' ' + slot
            for word in text[start:end].split():
                #if ('.' not in word) and ('?' not in word):
                word_to_slot[word] = slot
                #else:
                #    word_to_slot[word] = slot.replace('.','').replace('?','')
                if word[-1]=='.':
                    slot_texts = slot_texts + ' ' + word[:-1]
                else:
                    slot_texts = slot_texts + ' ' + word
        # Add context if it's there
        if 'context' in example:
            for req in example['context'].get('requestedSlots', []):
                text = req + " " + text

        # Create slots list
        slots = []
        cur = None
        for word in text.split():
            if ('.' in word) or ('?' in word):
                new_word = word.replace('.','').replace('?','')
            else:
                new_word = word
            if new_word in word_to_slot:
                slot = word_to_slot[new_word]
                if cur is not None and slot == cur:
                    slots.append("I-" + slot) 
                else:
                    slots.append("B-" + slot) 
                    cur = slot
            else:
                slots.append("O")
                cur = None
        return text, " ".join(slots), slot_texts 

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

class TOPDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 tokenizer: BertWordPieceTokenizer,
                 max_seq_length: int,
                 vocab_file_name: str,
                 prompt=[],
                 pet=None,
                 pet_final=''):
        # For caching
        data_dirname = os.path.dirname(os.path.abspath(data_path))
        split = os.path.basename(os.path.abspath(data_path))

        # Slot categories
        slot_vocab_path = os.path.join(os.path.dirname(data_path), "vocab.slot")
        slot_names = [e.strip() for e in open(slot_vocab_path).readlines()]
        slot_names.insert(0, "[PAD]")
        self.slot_label_to_idx = dict((label, idx) for idx, label in enumerate(slot_names))
        self.slot_idx_to_label = {idx: label for label, idx in self.slot_label_to_idx.items()}

        # Intent categories
        intent_vocab_path = os.path.join(data_dirname, "vocab.intent")
        intent_names = [e.strip() for e in open(intent_vocab_path).readlines()]
        self.intent_label_to_idx = dict((label, idx) for idx, label in enumerate(intent_names))
        self.intent_idx_to_label = {idx: label for label, idx in self.intent_label_to_idx.items()}

        # Process data
        self.tokenizer = tokenizer
        cached_path = os.path.join(data_dirname, "{}_{}_top_cached".format(split, vocab_file_name))
        if pet!=None:
            cached_path = cached_path + str(pet)
        if pet_final!='':
            cached_path = cached_path + 'pet'
        if prompt==['file1']:
            cached_path = cached_path + '_augmented'
        if not os.path.exists(cached_path):
            self.examples = []
            data = [e.strip() for e in open(data_path).readlines() ]
            for example in tqdm(data):
                example, intent = example.split(" <=> ")
                text_list = [e.split(":")[0] for e in example.split()]
                slot_list = [e.split(":")[1] for e in example.split()]
                text = " ".join(text_list)
                slots = " ".join(slot_list)
                slot_text = ''
                slot_value = ''
                for si in range(len(slot_list)):
                    if slot_list[si]!='':
                        if slot_list[si][0]=='B':
                            if slot_value != '':
                                if slot_text =='':
                                    slot_text = slot_name + ' ' + slot_value
                                else:
                                    slot_text = slot_text + ' ' + slot_name + ' ' + slot_value
                            slot_name = slot_list[si].split('SL')[-1].lower().replace('_',' ')#.replace('_', ' '), need to be modified to increase performance
                            # slot_name = slot_list[si].lower().replace('_',' ')
                            slot_value = text_list[si]
                        elif slot_list[si][0]=='I' :
                            slot_value = slot_value + ' ' + text_list[si]
                        elif slot_list[si][0]=='O':
                            if slot_value != '':
                                if slot_text =='':
                                    slot_text = slot_name + ' ' + slot_value
                                else:
                                    slot_text = slot_text + ' ' + slot_name + ' ' + slot_value
                            slot_value = ''
                if slot_value != '':
                    if slot_text =='':
                        slot_text = slot_name + ' ' + slot_value
                    else:
                        slot_text = slot_text + ' ' + slot_name + ' ' + slot_value
                
                # input
                def wrap_encoded(slot_encoded, intent_encoded, intent, slot_text, origin_input=None):
                    slot_attention=len(slot_encoded)*[1]
                    intent_attention=len(intent_encoded)*[1]
                    case = {
                    "ind": len(self.examples)
                    }
                    # if intent:
                    if intent[:2]=='IN':
                        encoded_intent = tokenizer.encode(intent[2:].replace('_',' ').lower())
                    else:
                        encoded_intent = tokenizer.encode(intent)
                    case["input_intent"] = np.array(intent_encoded)[-max_seq_length:] # name can be changed
                    case["attention_intent"] = np.array(intent_attention)[-max_seq_length:]
                    case["intent_ids"] = np.array(encoded_intent)[-max_seq_length:]
                    # if slot_text:
                    encoded_slot_text = tokenizer.encode(slot_text)
                    case["input_slot"] = np.array(slot_encoded)[-max_seq_length:]
                    case["attention_slot"] = np.array(slot_attention)[-max_seq_length:]
                    case["slot_ids"] = np.array(encoded_slot_text)[-max_seq_length:]
                    if origin_input:
                        case['origin_input'] = np.array(origin_input)[-max_seq_length:]
                    self.examples.append(case)
                    return
                
                if prompt ==['file1']:
                    if pet!=None:
                        intent_prompts = json.load(open('data/new_prompts/intent_pet.json','r'))[pet]
                        state_prompts = json.load(open('data/new_prompts/state_pet.json','r'))[pet]
                        if 'train' in split:
                            intent_ps = intent_prompts
                            slot_ps = state_prompts
                    else:
                        state_prompts = json.load(open('data/new_prompts/statesimplified_selected_dialoglue.json','r'))
                        intent_prompts = json.load(open('data/new_prompts/intentsimplified_selected_dialoglue.json','r'))
                        AUG_NUM = 25 # 30
                        if 'train' in split:
                            intent_ps = random.sample(intent_prompts, AUG_NUM)
                            slot_ps = random.sample(state_prompts, AUG_NUM)
                    TEST_NUM = 3
                    if 'val' in split:
                        intent_ps = random.sample(intent_prompts, 1)
                        slot_ps = random.sample(state_prompts, 1)
                    if 'test' in split:
                        intent_ps = random.sample(intent_prompts, TEST_NUM)
                        slot_ps = random.sample(state_prompts, TEST_NUM)
                    for num in range(len(intent_ps)):
                        intent_p = intent_ps[num]
                        slot_p = slot_ps[num]
                        p_text = ' ' + text + ('. ' if ('.' not in text and '?' not in text) else ' ')
                        intent_encoded = tokenizer.encode( intent_p[0] +p_text + intent_p[1])
                        slot_encoded = tokenizer.encode( slot_p[0] + p_text + slot_p[1])
                        # wrap_encoded(slot_encoded, [], None, slot_text, tokenizer.encode(text))
                        # wrap_encoded([], intent_encoded, intent, None, tokenizer.encode(text))
                        # independently train slot and intent, not available now
                        wrap_encoded(slot_encoded, intent_encoded, intent, slot_text) #, tokenizer.encode(text))
                else:
                    encoded = tokenizer.encode(text)
                    attention=len(encoded)*[1]
                    type_ids=(len(encoded)-1)*[1] + [0]
                    #output
                    intent_encoded = tokenizer.encode(intent[2:].replace('_',' ').lower())
                    intent_attention=len(intent_encoded)*[1]
                    encoded_slot_text = tokenizer.encode(slot_text)
                    encoded_slot_labels = self.encode_token_labels([text], [slots],
                                                                len(encoded),
                                                                tokenizer,
                                                                self.slot_label_to_idx,
                                                                max_seq_length)
                    # prompt: multi-task need to use prompt as task definition
                    intent_prompt = tokenizer.encode('What is the user intent?')[:-1] 
                    # please predict # without </s>
                    slot_prompt = tokenizer.encode('What is the user info?')[:-1]

                    self.examples.append({
                        "input_ids": np.array(encoded)[-max_seq_length:],
                        "attention_mask": np.array(attention)[-max_seq_length:],
                        "token_type_ids": np.array(type_ids)[-max_seq_length:],
                        "slot_labels": encoded_slot_labels[-max_seq_length:],
                        "intent_label": self.intent_label_to_idx[intent],
                        "intent_ids": np.array(intent_encoded)[-max_seq_length:],
                        "slot_ids": np.array(encoded_slot_text)[-max_seq_length:],
                        "intent_mask": np.array(intent_attention)[-max_seq_length:],
                        "intent_prompt": np.array(intent_prompt)[-max_seq_length:],
                        "slot_prompt": np.array(slot_prompt)[-max_seq_length:],
                        "ind": len(self.examples),
                    })
            if pet_final!='':
                PET_AUG_NUM = 10
                state_prompts = json.load(open('data/new_prompts/statesimplified_selected_dialoglue.json', 'r'))
                intent_prompts = json.load(open('data/new_prompts/intentsimplified_selected_dialoglue.json', 'r'))
                # pet_final shall be the intent path
                aug_data = json.load(open(pet_final,'r'))
                #aug_data_slot = json.load(open('dialoglue/pseudo_labels/pseudo_top_slot.json','r'))
                for aug in aug_data:
                    text = aug[0]
                    intent = aug[1]
                    slot_text = aug[2]
                    intent_ps = random.sample(intent_prompts, PET_AUG_NUM)
                    slot_ps = random.sample(state_prompts, PET_AUG_NUM)
                    for num in range(len(intent_ps)):
                        intent_p = intent_ps[num]
                        slot_p = slot_ps[num]
                        p_text = ' ' + text + ('. ' if ('.' not in text and '?' not in text) else ' ')
                        intent_encoded = tokenizer.encode( intent_p[0] +p_text + intent_p[1])
                        slot_encoded = tokenizer.encode( slot_p[0] + p_text + slot_p[1])
                        wrap_encoded(slot_encoded, intent_encoded, intent, slot_text)
                """
                for aug in aug_data_slot:
                    utt = aug[0]
                    slot_text = aug[1]
                    tmp_prompts = random.sample(state_prompts, PET_AUG_NUM)
                    for tmp_prompt in tmp_prompts:
                        encoded = tokenizer.encode(tmp_prompt[0] + ' ' + utt + tmp_prompt[1])
                        wrap_encoded(encoded, [], None, slot_text)
                """
            with open(cached_path, "wb") as f:
                pickle.dump(self.examples, f)
        else:
            LOGGER.info("Loading from cached path: {}".format(cached_path))
            with open(cached_path, "rb") as f:
                self.examples = pickle.load(f)

    def encode_token_labels(self,
                            text_sequences,
                            slot_names,
                            encoded_length,
                            tokenizer,
                            slot_map,
                            max_length) -> np.array:
        def _get_word_tokens_len(word: str, tokenizer: BertWordPieceTokenizer) -> int:
            return sum(map(lambda token: 1 if token not in tokenizer.all_special_tokens else 0,
                           tokenizer.tokenize(word)))
    
        encoded = np.zeros(shape=(len(text_sequences), encoded_length), dtype=np.int32)
        for i, (text_sequence, word_labels) in enumerate(
                zip(text_sequences, slot_names)):
            encoded_labels = []
            for word, word_label in zip(text_sequence.split(), word_labels.split()):
                encoded_labels.append(slot_map[word_label])
                expand_label = word_label.replace("B-", "I-")
                if not expand_label in slot_map:
                    expand_label = word_label
                word_tokens_len = _get_word_tokens_len(word, tokenizer)
                encoded_labels.extend([slot_map[expand_label]] * (word_tokens_len - 1))
            encoded[i, 1:len(encoded_labels) + 1] = encoded_labels
        return encoded.squeeze()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
