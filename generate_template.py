import copy
import random
import os
import argparse
#from typing import OrderedDict
import math
import attr

import transformers
import numpy as np
from transformers import T5ForConditionalGeneration, GPT2LMHeadModel, T5Tokenizer, GPT2Tokenizer, BertTokenizer
#from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import json
import argparse
import pandas as pd
from get_data import *
import get_data
from rouge_score import rouge_scorer

ALL_DOMAINS = ['intent', 'state', 'emotion', 'summary', 'answer', 'generated', 'response', 'choice', 'key information']
def generate(model, tokenizer, grounding, input, output, beam=10, case_num=50, no_extra=False, device=0):
    """
    Generate templates based on given inputs
    """
    torch.cuda.set_device(device)
    return_num = 5
    input_texts = []
    input_tensors = []
    max_length = 0
    pre_prompts = []
    prompts = []
    seq = {}
    eos_id = tokenizer.convert_tokens_to_ids('<extra_id_3>')
    split0 = tokenizer.convert_tokens_to_ids('<extra_id_0>')
    split1 = tokenizer.convert_tokens_to_ids('<extra_id_1>')
    split2 = tokenizer.convert_tokens_to_ids('<extra_id_2>')
    random_case = random.choices(range(len(input)),k=case_num)
    with torch.no_grad():
        for i in random_case:
            if grounding != []:
        # Process the inputs
                seq[0] = '<extra_id_0>:' + grounding[i] + '<extra_id_1>:' + input[i] + '<extra_id_2>:' + output[i]
                seq[1] = '<extra_id_0>' + grounding[i] + '<extra_id_1>:' + input[i] + '<extra_id_2>:' + output[i]
                seq[2] = '<extra_id_0>' + grounding[i] + '<extra_id_1>' + input[i] + '<extra_id_2>:' + output[i]
                seq[3] = '<extra_id_0>' + grounding[i] + '<extra_id_1>' + input[i] + '<extra_id_2>' + output[i]
                seq[4] = '<extra_id_0>:' + grounding[i] + '<extra_id_1>' + input[i] + '<extra_id_2>:' + output[i]
            else:
                seq[0] = ['<extra_id_0>:' + input[i] + '<extra_id_1>:' + output[i]]
                seq[1] = ['<extra_id_0>' + input[i] + '<extra_id_1>:' + output[i]]
                seq[2] = ['<extra_id_0>:' + input[i] + '<extra_id_1>' + output[i]]
                seq[3] = ['<extra_id_0>' + input[i] + '<extra_id_1>' + output[i]]
            if no_extra:
                seq = {}
                seq[3] = '<extra_id_0>' + grounding[i] + '<extra_id_1>' + input[i] + '<extra_id_2>' + output[i]
                seq[5] = grounding[i] + '<extra_id_0>' + input[i] + '<extra_id_1>' + output[i]
            if input[i] == '':
                seq = {}
                seq[6] = '<extra_id_0>' + grounding[i] + '<extra_id_1>' + output[i]
                seq[7] = grounding[i] + '<extra_id_0>' + output[i]
            for type,se in seq.items():
                if len(se)<1024:
                    tokenized = tokenizer(se, return_tensors="pt")
                    input_ids = tokenized.input_ids.cuda()
                    attention_mask = tokenized.attention_mask.cuda()
                    template = model.generate(input_ids=input_ids, attention_mask = attention_mask, eos_token_id=eos_id,num_beams=beam, num_return_sequences=return_num, no_repeat_ngram_size=1, encoder_no_repeat_ngram_size=2) #no_repeat_ngram_size=2., encoder_no_repeat_ngram_size=3     
                    for j in range(return_num):
                        r = template.cpu().tolist()[j]
                        if grounding != []:
                            final = []
                            if type==5 :
                                if split1 in r:
                                    input_prompt = tokenizer.decode(r[:r.index(split1)], skip_special_tokens=True)
                                    output_prompt = tokenizer.decode(r[r.index(split1):], skip_special_tokens=True)
                                    final = [input_prompt.replace('-',''), output_prompt.replace('-','')]
                                else:
                                    print('no split1 error')
                                    print(tokenizer.decode(r, skip_special_tokens=True))
                            elif type==6 :
                                if split1 in r:
                                    input_prompt = tokenizer.decode(r[:r.index(split1)], skip_special_tokens=True)
                                    output_prompt = tokenizer.decode(r[r.index(split1):], skip_special_tokens=True)
                                    final = [input_prompt.replace('-',''), '', output_prompt.replace('-','')]
                                else:
                                    print('no split1 error')
                                    print(tokenizer.decode(r, skip_special_tokens=True))
                            elif type==7 :
                                if split0 in r:
                                    input_prompt = tokenizer.decode(r[:r.index(split0)], skip_special_tokens=True)
                                    final = ['', input_prompt]
                                else:
                                    print('no split0 error')
                            elif (split1 in r) and (split2 in r): 
                                grounding_prompt = tokenizer.decode(r[:r.index(split1)], skip_special_tokens=True)
                                input_prompt = tokenizer.decode(r[r.index(split1):r.index(split2)], skip_special_tokens=True)
                                output_prompt = tokenizer.decode(r[r.index(split2):], skip_special_tokens=True)
                                if type==0:
                                    final = [grounding_prompt+':', input_prompt+':', output_prompt+':']
                                elif type==1:
                                    final = [grounding_prompt.replace('-',''), input_prompt+':', output_prompt+':']
                                elif type==2:
                                    final = [grounding_prompt.replace('-',''), input_prompt.replace('-',''), output_prompt+':']
                                elif type==3:
                                    final = [grounding_prompt.replace('-',''), input_prompt.replace('-',''), output_prompt.replace('-','')]
                                elif type==4:
                                    final = [grounding_prompt+':', input_prompt.replace('-',''), output_prompt+':']
                            if (final!=[]) and (final not in prompts):
                                if final not in pre_prompts:
                                    pre_prompts.append(final)
                                else:
                                    prompts.append(final)
                        else:
                            if split1 in r :
                                input_prompt = tokenizer.decode(r[:r.index(split1)], skip_special_tokens=True)
                                output_prompt = tokenizer.decode(r[r.index(split1):], skip_special_tokens=True)
                                if type==0:
                                    final = [input_prompt+':', output_prompt+':']
                                elif type==1:
                                    final = [input_prompt.replace('-',''), output_prompt+':']
                                elif type==2:
                                    final = [input_prompt+':', output_prompt.replace('-','')]
                                elif type==3:
                                    final = [input_prompt.replace('-',''), output_prompt.replace('-','')]
                                if final not in prompts:
                                    if final not in pre_prompts:
                                        pre_prompts.append(final)
                                    else:
                                        prompts.append(final)
    return prompts

def load_dataset(task,task_num=None):
    func_name = 'get_' + task
    for num in [0,1]:
        if str(num) in task:
            task_num = num
            func_name = func_name.replace(str(num),'')
    k = getattr(get_data, func_name)
    if task_num:
        return k(task_num)
    else:
        return k()
"""
def collate_fn(self,tokenizer = tokenizer):
    pad_id = tokenizer.pad_token_id
    pad_result = {}
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
"""
def evaluate(model, tokenizer, seq, output, eval_num=512, batch_size=16):
    score = []
    eos_id = tokenizer.convert_tokens_to_ids('</s>')
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    #evaluator = rough.Rouge(metrics='rouge-2')
    acc = 0
    metrics = 0
    with torch.no_grad():
        for i in range(int(eval_num/batch_size)):
            tokenized = tokenizer(seq[batch_size*i:batch_size*(i+1)], return_tensors="pt", padding=True, max_length=512) # truncation = True 
            input_ids = tokenized.input_ids.cuda()
            attention_mask = tokenized.attention_mask.cuda()
            output_token = model.generate(input_ids=input_ids, attention_mask = attention_mask, eos_token_id=eos_id)
            output_g = output_token.cpu().tolist()
            for j in range(batch_size):
                output_gen = tokenizer.decode(output_g[j], skip_special_tokens=True)
                if output_gen == output[batch_size*i+j]:
                    acc = acc +1
                rouge = scorer.score(output_gen, output[i])
                metrics = metrics + rouge['rouge1'][2] + rouge['rougeL'][2]
                
        acc = acc/eval_num
        metrics = metrics/eval_num
        score = acc + metrics
        return score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--t5_model', type=str, default='t5-large', help='T5 pre-trained model') # t5-large
    parser.add_argument('--seed', type=int, nargs='+', default=[42, 13, 21, 100, 87], help="Data split seeds")
    parser.add_argument('--task_name', type=str, nargs='+', default=['squad','quac','narrativeqa','race','doqa',#doc_qa,
    'dream','dialog_re', 'ddrel', 'friends_qa','molweni','mutual', #
    'sparc','spider', #text2sql,
    'dailydialog','personachat','metawoz','empathetic_dialogues',#chat,
    'samsum', 'dialogsum', 'reading_comprehension',#summary 
    'sentihood', 'aste', 'mams',# slot_filling  
    'commonsense_qa', 'cosmos_qa',# commonsense qa  
    'go_emotions', 'meld', 'reccon',# emotional conversation 
    'kgdial', 'woi', 'commonsense_dialog',# kgdial  
    'xlsum', 'web_nlg', 'xwikis','wiki_lingua', 'dart', 'e2e_nlg', 'common_gen' # generation, 'totto',
    'ketod', 'fusedchat'], #'fused_chat'
    help="Task names")
    parser.add_argument('--output_dir', type=str, default='auto_template/generated')

    parser.add_argument('--data_dir', type=str, default="data/k-shot", help="Data directory")
    parser.add_argument('--beam', type=int, default=10, help="Beam search width")
    parser.add_argument('--k', type=int, default=16, help="Number of training instances per label")
 
    args = parser.parse_args()

    model = T5ForConditionalGeneration.from_pretrained(args.t5_model)
    tokenizer = T5Tokenizer.from_pretrained(args.t5_model)
    #tokenizer.sep_token = '</s>'

    model = model.cuda()
    model.eval()
    for task_name in args.task_name:
        prompt_dict = {}
        prompt_dict_t0 = {}
        grounding, query, answer,templated_data, task_description= load_dataset(task_name)
        #for seed in args.seed:
        # eval t0
        if templated_data!={}:
            for t1,data_set in templated_data.items():
                s = evaluate(model=model, tokenizer=tokenizer, seq=list(t[0] for t in data_set), output=list(t[1] for t in data_set), eval_num=1024)
                prompt_dict_t0[t1] = s
        td_input = []
        td_input_ahead = []
        no_prompt = []
        #task description and no prompt
        for i in range(len(query)):
            if grounding != []:
                td_input.append((grounding[i] + ' ' + query[i] + ' '+task_description))
                td_input_ahead.append((task_description + ' ' + grounding[i] + ' ' + query[i]))
                no_prompt.append((grounding[i] + ' ' + query[i]))
            else:
                td_input.append((query[i] + ' '+task_description))
                td_input_ahead.append((task_description + ' ' + query[i] ))
                no_prompt.append(query[i])
        s = evaluate(model=model, tokenizer=tokenizer, seq=td_input, output=answer, eval_num=(1024 if len(query)>1024 else 512))
        s1 = evaluate(model=model, tokenizer=tokenizer, seq=no_prompt, output=answer, eval_num=(1024 if len(query)>1024 else 512))
        s2 = evaluate(model=model, tokenizer=tokenizer, seq=td_input_ahead, output=answer, eval_num=(1024 if len(query)>1024 else 512))
        prompt_dict_t0['task description'] = s
        prompt_dict_t0['no prompt'] = s1
        prompt_dict_t0['task description ahead'] = s2
        json.dump(prompt_dict_t0,open(('data/prompts/' + task_name + '_t0.json'), 'w'), indent=2)
        # eval generated template
        template = generate(model=model, tokenizer=tokenizer, grounding=grounding, input=query, output=answer, beam=args.beam)
        scores = []
        for t in template:
            seq =[]
            for i in range(len(query)):
                if isinstance(t, list):
                    if grounding != []:
                        seq.append(t[0] + ' '+ grounding[i] + ' '+t[1] + ' '+ query[i] + ' '+ t[2])
                    else:
                        seq.append( t[0] + ' '+ query[i] + ' '+ t[1])
                else:
                    if grounding != []:
                        seq.append(grounding[i] + ' ' + query[i] + ' ' + t.replace('_',' '))
                    else:
                        seq.append(+ query[i] + t.replace('_',' '))
            scores.append(evaluate(model=model, tokenizer=tokenizer, seq=seq, output=answer))
        origin_scores = copy.deepcopy(scores)
        scores.sort(reverse=True)
        for score in scores:
            prompt = template[origin_scores.index(score)]
            prompt_dict[score] =prompt
        json.dump(prompt_dict,open(('data/prompts/' + task_name + '.json'), 'w'), indent=2)

def collect_prompt_result():
    all_task_name = ['squad','quac','narrativeqa','race','doqa',#doc_qa#dialqa,
    'dream','dialog_re', 'ddrel', 'friends_qa','molweni','mutual', #
    'spider', 'sparc', #text2sql,
    'dailydialog','personachat','metawoz','empathetic_dialogues',#chat,
    'samsum', 'dialogsum', 'reading_comprehension',#summary 
    'sentihood', 'aste', 'mams',# slot_filling  
    'commonsense_qa', 'cosmos_qa',# commonsense qa  
    'go_emotions', 'meld', 'reccon',# emotional conversation 
    'kgdial', 'woi', 'commonsense_dialog',# kgdial  
    'xlsum', 'web_nlg', 'xwikis','wiki_lingua', 'dart', 'e2e_nlg', 'common_gen',#nlg
    'ketod', 'fusedchat']#fused_chat
    all_prompts = {}
    multitask = ['woi','dailydialog','reccon','meld','ketod']
    case_num = 0
    for task_name in all_task_name:
        grounding, query, answer,templated_data, task_description= load_dataset(task_name)
        case_num = case_num + len(query)
        data = [grounding,query,answer]

        if task_name in multitask:
            json.dump(data,open(('data/pre_train/prompts/' + task_name + '0.json'), 'w'), indent=2)
            grounding, query, answer,templated_data, task_description= load_dataset(task_name,task_num = 1)
            json.dump([grounding, query, answer],open(('data/pre_train/prompts/' + task_name + '1.json'), 'w'), indent=2)
        else:
            json.dump(data,open(('data/pre_train/prompts/' + task_name + '.json'), 'w'), indent=2)

        prompts =[]
        prompt_dir = ['data/prompts','data/prompts_unrestricted'] 
        dir = 'data/prompts_unrestricted'
        t0_file = os.path.join(dir,(task_name+'_t0.json'))
        t0_prompts = json.loads(open(t0_file,'r').read())
        td,np, tda = t0_prompts['task description'],t0_prompts['task description ahead'],t0_prompts['no prompt']
        score = max(td,np,tda)
        if td> np and td>tda:
            prompts.append('task description,'+task_description)
        elif np> tda :
            prompts.append('no prompt')
        else:
            prompts.append('task description ahead,'+task_description)
        for name,s in t0_prompts.items():
            if name not in ['task description','task description ahead','no prompt']:
                if s > score :
                    prompts.append(name)  
        for dir in prompt_dir:                
            filename = os.path.join(dir,(task_name+'.json'),)
            gen_prompts = json.loads(open(filename,'r').read(),object_pairs_hook=OrderedDict)
            for k in list(gen_prompts.keys())[:5]:
                v = gen_prompts[k]
                if float(k)>score:
                    prompts.append(v)
        all_prompts[task_name] = prompts
    json.dump(all_prompts,open(('data/all_prompts.json'), 'w'), indent=2)
    return 

def get_intent():
    #file = 'data/pre_train/prompts/space_single_intent_glue.json' # _glue
    file = 'data/pre_train/space_single_intent.json'
    data = json.load(open(file,'r'))
    grounding = []
    new_grounding = []
    answers = []
    new_answers = []
    for dataset in data:
        grounding.extend(dataset[0])
        answers.extend(dataset[2])
    shuf = list(range(len(grounding)))
    random.shuffle(shuf)
    for i in shuf:
        new_grounding.append(grounding[i])
        new_answers.append(answers[i])
    return new_grounding,new_answers

def get_state():
    file = 'data/pre_train/space_single_state.json'
    data = json.load(open(file,'r'))
    grounding = []
    new_grounding = []
    answers = []
    new_answers = []
    for dataset in data:
        grounding.extend(dataset[0])
        answers.extend(dataset[2])
    shuf = list(range(len(grounding)))
    random.shuffle(shuf)
    for i in shuf:
        new_grounding.append(grounding[i])
        new_answers.append(answers[i])
    return new_grounding,new_answers

def get_emotion():
    names = json.load(open('data/pre_train/prompts/all_tasks.json','r'))['emotional']
    grounding = []
    answers = []
    for name in names:
        g, q, a, _, _ = load_dataset(name)
        if g != []:
            grounding.extend(g)
        else:
            grounding.extend(q)
        answers.extend(a)
    new_grounding = []
    new_answers = []
    shuf = list(range(len(grounding)))
    random.shuffle(shuf)
    for i in shuf:
        if answers[i]!= 'no emotion':
            new_grounding.append(grounding[i])
            new_answers.append(answers[i])
    return new_grounding,new_answers

def get_summary():
    names = json.load(open('data/pre_train/prompts/all_tasks.json','r'))['summary']
    # samsum, dialogsum, siki_lingua
    grounding = []
    answers = []
    for name in names:
        g, q, a, _, _ = load_dataset(name)
        if g != []:
            grounding.extend(g)
        else:
            grounding.extend(q)
        answers.extend(a)
    new_grounding = []
    new_answers = []
    shuf = list(range(len(grounding)))
    random.shuffle(shuf)
    for i in shuf:
        if len(grounding[i])<1200:
            new_grounding.append(grounding[i])
            new_answers.append(answers[i])
    return new_grounding,new_answers

def get_qa():
    names = json.load(open('data/pre_train/prompts/all_tasks.json','r'))['doc_qa']
    names.extend(json.load(open('data/pre_train/prompts/all_tasks.json','r'))['dialqa'])
    grounding = []
    answers = []
    for name in names:
        if name not in []:
            g, q, a, _, _ = load_dataset(name)
            for i in range(len(a)):
                if g != []:
                    grounding.append(g[i] + ' ' + q[i])
                else:
                    grounding.append(q[i])
                answers.append(a[i])
    new_grounding = []
    new_answers = []
    shuf = list(range(len(grounding)))
    random.shuffle(shuf)
    for i in shuf:
        if len(grounding[i])<1200:
            new_grounding.append(grounding[i])
            new_answers.append(answers[i])
    return new_grounding,new_answers

def get_nlg():
    names = json.load(open('data/pre_train/prompts/all_tasks.json','r'))['nlg']
    grounding = []
    answers = []
    for name in names:
        g, q, a, _, _ = load_dataset(name)
        for i in range(len(a)):
            if g != []:
                grounding.append(g[i] + ' ' + q[i])
            else:
                grounding.append(q[i])
            answers.append(a[i])
        answers.extend(a)
    new_grounding = []
    new_answers = []
    shuf = list(range(len(grounding)))
    random.shuffle(shuf)
    for i in shuf:
        if len(grounding[i])<1200:
            new_grounding.append(grounding[i])
            new_answers.append(answers[i])
    return new_grounding,new_answers

def get_response():
    names = json.load(open('data/pre_train/prompts/all_tasks.json','r'))['response_generation']
    grounding = []
    answers = []
    for name in names:
        g, q, a, _, _ = load_dataset(name)
        for i in range(len(a)):
            if g != []:
                grounding.append(g[i] + ' ' + q[i])
            else:
                grounding.append(q[i])
            answers.append(a[i])
    new_grounding = []
    new_answers = []
    shuf = list(range(len(grounding)))
    random.shuffle(shuf)
    for i in shuf:
        if len(grounding[i])<1200:
            new_grounding.append(grounding[i])
            new_answers.append(answers[i])
    return new_grounding,new_answers

def get_choice():
    names = json.load(open('data/pre_train/prompts/all_tasks.json','r'))['multiple_choice']
    # samsum, dialogsum, siki_lingua
    grounding = []
    answers = []
    for name in names:
        g, q, a, _, _ = load_dataset(name)
        for i in range(len(a)):
            if g != []:
                grounding.append(g[i] + ' ' + q[i])
            else:
                grounding.append(q[i])
            answers.append(a[i])
    new_grounding = []
    new_answers = []
    shuf = list(range(len(grounding)))
    random.shuffle(shuf)
    for i in shuf:
        if len(grounding[i])<1200:
            new_grounding.append(grounding[i])
            new_answers.append(answers[i])
    return new_grounding,new_answers

def get_txt2sql():
    names = json.load(open('data/pre_train/prompts/all_tasks.json','r'))['text2sql']
    # samsum, dialogsum, siki_lingua
    grounding = []
    answers = []
    for name in names:
        g, q, a, _, _ = load_dataset(name)
        for i in range(len(a)):
            if g != []:
                grounding.append(g[i] + ' ' + q[i])
            else:
                grounding.append(q[i])
            answers.append(a[i])
    new_grounding = []
    new_answers = []
    shuf = list(range(len(grounding)))
    random.shuffle(shuf)
    for i in shuf:
        if len(grounding[i])<1200:
            new_grounding.append(grounding[i])
            new_answers.append(answers[i])
    return new_grounding,new_answers

def get_summary_zh():
    names = ['CSDS_SFT', '20230206_tingwu123_sft']
    # samsum, dialogsum, siki_lingua
    grounding = []
    answers = []
    for name in names:
        g = []  #:list of cases, data[0]['3842']['apprentice_persona']# remove '\n'  #['dialog_history]:list of turns: 'action','text', 'context'
        q = []
        a = []
        tmp = '1'
        with open(os.path.join(name, 'train.json'), "r", encoding="utf-8") as fp:  #8614 dialog,167667turns
            while tmp != "":
                tmp = fp.readline()
                if tmp != '':
                    item = json.loads(tmp)
                    q.append(item['Dialogue'] if 'Dialogue' in item else item['source'])
                    a.append(item['FinalSumm'] if 'FinalSumm' in item else item['target'])
        for i in range(len(a)):
            if g != []:
                grounding.append(g[i] + ' ' + q[i])
            else:
                grounding.append(q[i])
            answers.append(a[i])
    new_grounding = []
    new_answers = []
    shuf = list(range(len(grounding)))
    random.shuffle(shuf)
    for i in shuf:
        if len(grounding[i])<1200:
            new_grounding.append(grounding[i])
            new_answers.append(answers[i])
    return new_grounding,new_answers

def get_nlu_zh():
    names = ['zh_data/nlu-intent_classification-diaoxiaomi-for-prompt-5k.txt',
    'zh_data/滴滴.csv']
    # samsum, dialogsum, siki_lingua
    grounding = []
    answers = []
    for name in names:
        g = []  #:list of cases, data[0]['3842']['apprentice_persona']# remove '\n'  #['dialog_history]:list of turns: 'action','text', 'context'
        q = []
        a = []
        tmp = '1'
        with open(name, "r", encoding="utf-8") as fp:  #8614 dialog,167667turns
            while tmp != "":
                tmp = fp.readline()
                if tmp != '':
                    if '\t' in tmp:
                        item = tmp.split('\t')
                        if '?' not in item[0]:
                            q.append(item[0]+'?')
                        else:
                            q.append(item[0])
                        a.append(item[1].replace('\n', ''))
                    else:
                        item = tmp.split(',')
                        if '?' not in item[0]:
                            q.append(item[0]+'?')
                        else:
                            q.append(item[0])
                        a.append(item[1].replace('\n', ''))
                    #q.append(item[0] if 'Dialogue' in item else item['source'])
                    #a.append(item['FinalSumm'] if 'FinalSumm' in item else item['target'])
        for i in range(len(a)):
            if g != []:
                grounding.append(g[i] + ' ' + q[i])
            else:
                grounding.append(q[i])
            answers.append(a[i])
    new_grounding = []
    new_answers = []
    shuf = list(range(len(grounding)))
    random.shuffle(shuf)
    for i in shuf:
        if len(grounding[i])<1200:
            new_grounding.append(grounding[i])
            new_answers.append(answers[i])
    return new_grounding,new_answers

def get_sql_zh():
    names = ['zh_data/sql-to-response-for-prompt.txt']
    # samsum, dialogsum, siki_lingua
    grounding = []
    answers = []
    for name in names:
        g = []  #:list of cases, data[0]['3842']['apprentice_persona']# remove '\n'  #['dialog_history]:list of turns: 'action','text', 'context'
        q = []
        a = []
        tmp = '1'
        with open(name, "r", encoding="utf-8") as fp:  #8614 dialog,167667turns
            while tmp != "":
                tmp = fp.readline()
                if tmp != '':
                    if '\t' in tmp:
                        item = tmp.split('\t')
                        #if '?' not in item[0]:
                        #q.append(item[0]+'?')
                        #else:
                        q.append(item[0])
                        a.append(item[1].replace('\n', ''))
        for i in range(len(a)):
            if g != []:
                grounding.append(g[i] + ' ' + q[i])
            else:
                grounding.append(q[i])
            answers.append(a[i])
    new_grounding = []
    new_answers = []
    shuf = list(range(len(grounding)))
    random.shuffle(shuf)
    for i in shuf:
        if len(grounding[i])<1200:
            new_grounding.append(grounding[i])
            new_answers.append(answers[i])
    return new_grounding,new_answers

def get_act_zh():
    names = ['zh_data/policy_classification-clouldbee-for-prompt-5k.txt']
    # samsum, dialogsum, siki_lingua
    grounding = []
    answers = []
    for name in names:
        g = []  #:list of cases, data[0]['3842']['apprentice_persona']# remove '\n'  #['dialog_history]:list of turns: 'action','text', 'context'
        q = []
        a = []
        tmp = '1'
        with open(name, "r", encoding="utf-8") as fp:  #8614 dialog,167667turns
            while tmp != "":
                tmp = fp.readline()
                if tmp != '':
                    if '\t' in tmp:
                        item = tmp.split('\t')
                        #if '?' not in item[0]:
                        #q.append(item[0]+'?')
                        #else:
                        q.append(item[0])
                        a.append(item[1].replace('\n', ''))
        for i in range(len(a)):
            if g != []:
                grounding.append(g[i] + ' ' + q[i])
            else:
                grounding.append(q[i])
            answers.append(a[i])
    new_grounding = []
    new_answers = []
    shuf = list(range(len(grounding)))
    random.shuffle(shuf)
    for i in shuf:
        if len(grounding[i])<1200:
            new_grounding.append(grounding[i])
            new_answers.append(answers[i])
    return new_grounding,new_answers

def get_domain_zh():
    names = ['zh_data/nlu-session_classification-for-prompt-5k.txt']
    # samsum, dialogsum, siki_lingua
    grounding = []
    answers = []
    for name in names:
        g = []  #:list of cases, data[0]['3842']['apprentice_persona']# remove '\n'  #['dialog_history]:list of turns: 'action','text', 'context'
        q = []
        a = []
        tmp = '1'
        with open(name, "r", encoding="utf-8") as fp:  #8614 dialog,167667turns
            while tmp != "":
                tmp = fp.readline()
                if tmp != '':
                    if '\t' in tmp:
                        item = tmp.split('\t')
                        #if '?' not in item[0]:
                        #q.append(item[0]+'?')
                        #else:
                        if item[-1]!='-1\n':
                            q.append(''.join(item[-11:-2]))
                            a.append(item[-1].split('_')[0])
        for i in range(len(a)):
            if g != []:
                grounding.append(g[i] + ' ' + q[i])
            else:
                grounding.append(q[i])
            answers.append(a[i])
    new_grounding = []
    new_answers = []
    shuf = list(range(len(grounding)))
    random.shuffle(shuf)
    for i in shuf:
        if grounding[i]!='' and answers[i]!='':
        #if len(grounding[i])<480:
            new_grounding.append(grounding[i][-480:])
            new_answers.append(answers[i])
    return new_grounding,new_answers

def get_emotion_zh():
    names = ['zh_data/Sentiment-train-for-prompt-6k.txt']
    # samsum, dialogsum, siki_lingua
    grounding = []
    answers = []
    for name in names:
        g = []  
        q = []
        a = []
        tmp = '1'
        with open(name, "r", encoding="utf-8") as fp:  
            while tmp != "":
                tmp = fp.readline()
                if tmp != '':
                    if '\t' in tmp:
                        item = tmp.split('\t')
                        #if '?' not in item[0]:
                        #q.append(item[0]+'?')
                        #else:
                        q.append(item[0])
                        a.append(item[-1].replace('\n', ''))
        for i in range(len(a)):
            if g != []:
                grounding.append(g[i] + ' ' + q[i])
            else:
                grounding.append(q[i])
            answers.append(a[i])
    new_grounding = []
    new_answers = []
    shuf = list(range(len(grounding)))
    random.shuffle(shuf)
    for i in shuf:
        if len(grounding[i])<1200:
            new_grounding.append(grounding[i])
            new_answers.append(answers[i])
    return new_grounding,new_answers

def get_similarity_zh():
    names = ['zh_data/STS-AFQMC-train-from-prompt.json']
    # samsum, dialogsum, siki_lingua
    grounding = []
    answers = []
    label_switch = {'0':'不一致', '1':'一致'}
    for name in names:
        g = []  #:list of cases, data[0]['3842']['apprentice_persona']# remove '\n'  #['dialog_history]:list of turns: 'action','text', 'context'
        q = []
        a = []
        tmp = '1'
        with open(name, "r", encoding="utf-8") as fp:  #8614 dialog,167667turns
            while tmp != "":
                tmp = fp.readline()
                if tmp != '':
                    item = json.loads(tmp)
                    q.append('“' + item['sentence1'] + '”和“' + item['sentence2'] + '”')
                    a.append(label_switch[item['label']])
        for i in range(len(a)):
            if g != []:
                grounding.append(g[i] + ' ' + q[i])
            else:
                grounding.append(q[i])
            answers.append(a[i])
    new_grounding = []
    new_answers = []
    shuf = list(range(len(grounding)))
    random.shuffle(shuf)
    for i in shuf:
        if len(grounding[i])<1200:
            new_grounding.append(grounding[i])
            new_answers.append(answers[i])
    return new_grounding,new_answers

def get_dst_zh():
    names = ['zh_data/info-collect-dst-for-prompt.txt']
    # samsum, dialogsum, siki_lingua
    grounding = []
    answers = []
    for name in names:
        g = []  #:list of cases, data[0]['3842']['apprentice_persona']# remove '\n'  #['dialog_history]:list of turns: 'action','text', 'context'
        q = []
        a = []
        tmp = '1'
        with open(name, "r", encoding="utf-8") as fp:  #8614 dialog,167667turns
            while tmp != "":
                tmp = fp.readline()
                if tmp != '':
                    if '\t' in tmp:
                        item = tmp.split('\t')
                        #if '?' not in item[0]:
                        #q.append(item[0]+'?')
                        #else:
                        if '[SEP]' in item[0]:
                            hist = item[0].split('[SEP]')[0]
                            hist = item[0].split('[SEP]')[1]
                        else:
                            hist = ''
                            hist = item[0]
                        q.append(item[0])
                        a.append(item[1].replace('\n', ''))
        for i in range(len(a)):
            if g != []:
                grounding.append(g[i] + ' ' + q[i])
            else:
                grounding.append(q[i])
            answers.append(a[i])
    new_grounding = []
    new_answers = []
    shuf = list(range(len(grounding)))
    random.shuffle(shuf)
    for i in shuf:
        if len(grounding[i])<1200:
            new_grounding.append(grounding[i])
            new_answers.append(answers[i])
    return new_grounding,new_answers

def generate_template_using_keyword(domain):
    get_domain_data = {'intent':get_intent, 'state':get_state,
    'emotion':get_emotion, 'summary':get_summary, 'answer':get_qa, 'generated':get_nlg
    , 'response':get_response, 'choice':get_choice, 'key information':get_txt2sql,
    'summary_zh':get_summary_zh, 'nlu_zh':get_nlu_zh, 'sql_zh':get_sql_zh,
    'act_zh':get_act_zh, 'dst_zh':get_dst_zh, 'domain_zh':get_domain_zh, 'emotion_zh':get_emotion_zh, 
    'similarity_zh':get_similarity_zh}
    #grounding, answers = get_intent()
    grounding, answers = get_domain_data[domain]()
    device=0
    torch.cuda.set_device(device)
    keyword_dict ={'intent':['user intent', 'user intention', "user's purpose", "user's purpose", "user's intention", 'user purpose', ''], 
    'state': ['user state', "user's state", 'user info', 'user information', 'user status', ''], 
    'emotion':['emotion', 'sentiment', 'feeling', 'feels', ''], 
    'summary':['summary', 'sum up', 'main idea', 'outline', ''], 
    'answer':['Answer', 'A:', 'answer', 'Response', 'reponse', 'respond', 'reply', ''], 
    'generated':['generate', 'Generate', 'expression', 'express', 'keywords', ''], 
    'response':['Response', 'reponse', 'respond', 'reply', ''],
    'choice':['Choose', 'choose', 'choice', 'Answer', 'A:', 'answer', ''],
    'key information':['information', 'format', 'change', 'structured language', 'structure', ''],
    'summary_zh':['对话摘要', '摘要', '重点', '主要内容', '要点', '简述',
    '概括', '概述', '总结', ''],
    'nlu_zh':['用户意图', '客户意图',
    '意图', '目的', '问题', '意思', '场景', '方面'],
    'sql_zh':['答复', '数据库查询结果', '重新表述', '语言', '程序'
    , '解释', '说明', '说明注释', '查询结果', '通俗的语言', '重新整理'
    , '整理一下', '整理'],
    'emotion_zh':['情感', '情绪', '态度', '心理', '立场'
    , '心态'],
    'similarity_zh':['相似', '一致', '意思', '一样', '类似'
    , '像', '相同', '同样', '同样的意思'],
    'domain_zh':['事件类别', '对话主题', '类别', '主题', '领域'
    , '类型', '主旨', '题材', '焦点'],
    'act_zh':['策略', '对话动作', '方法', '对策', '方案'
    , '方式', '技巧', '做法'],
    'dst_zh':['事件类别', '对话主题', '类别', '主题', '领域'
    , '类型', '主旨', '题材', '焦点']}
    # new key word for zh tasks
    for adj in ['对话','演讲','会议''讨论','口语']:
        for key in ['摘要','主要内容', '大体内容', '主要意思']:
            keyword_dict['summary_zh'].append(adj +'的' + key)
    for adj in ['来电', '询问', '用户', '客户']:
        for key in ['意图','目的', '意思']:
            keyword_dict['nlu_zh'].append(adj +'的' + key)
    for adj in ['这句话']:
        for key in ['立场', '情感', '态度']:
            keyword_dict['emotion_zh'].append(adj +'的' + key)
    for adj in ['这两句话', '这两句话的意思']:
        for key in ['一致','相似', '一样', '类似', '相同']:
            keyword_dict['similarity_zh'].append(adj +'' + key) 
    for adj in ['事件', '对话', '电话', '客服']:
        for key in ['类别','领域', '主题', '焦点']:
            keyword_dict['domain_zh'].append(adj +'的' + key) 
    for adj in ['回复', '客服', '对话', '回答']:
        for key in ['策略', '方法', '方案', '对策', '方式']:
            keyword_dict['act_zh'].append(adj +'的' + key)

    keywords = keyword_dict[domain]

    if 'zh' in domain:
        gpt_model = GPT2LMHeadModel.from_pretrained('uer/gpt2-chinese-cluecorpussmall')
        gpt_tokenizer = BertTokenizer.from_pretrained('uer/gpt2-chinese-cluecorpussmall')
        gpt_model = gpt_model.cuda()
        gpt_model.eval()
        t5_tokenizer = BertTokenizer.from_pretrained('IDEA-CCNL/Randeng-T5-Char-700M-Chinese', add_special_tokens=False)
        t5_model = T5ForConditionalGeneration.from_pretrained('IDEA-CCNL/Randeng-T5-Char-700M-Chinese')
        t5_model.eval()
    else:
        gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')
        gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        gpt_model = gpt_model.cuda()
        gpt_model.eval()
        t5_model = T5ForConditionalGeneration.from_pretrained('t5-large')
        t5_tokenizer = T5Tokenizer.from_pretrained('t5-large')
        t5_model.eval()
    #tokenizer=BertTokenizer.from_pretrained('IDEA-CCNL/Randeng-T5-Char-700M-Chinese', add_special_tokens=False)
    #model=T5ForConditionalGeneration.from_pretrained('IDEA-CCNL/Randeng-T5-Char-700M-Chinese')
    t5_model = t5_model.cuda()
    t5_model.eval()
    prompt_dict = {}
    template = {}
    #scores = {}
    seq = []
    for i in range(len(grounding)):
        #seq.append( (grounding[i] + ' What is the user intent? ')) # baseline for intent
        if 'zh' in domain:
            seq.append( (grounding[i] + keyword_dict[domain][0]  + '是： '+ answers[i]))
        else:
            seq.append( (grounding[i] + ' What is the ' + domain  + ' ? '+ answers[i]))
    #prompt_dict['base'] = evaluate(model=model, tokenizer=tokenizer, seq=seq, output=answers, eval_num=2048, batch_size=64)
    prompt_dict['base'] = get_ppl(model=gpt_model, tokenizer=gpt_tokenizer, seq=seq, eval_num=4096, batch_size=16,device=device)
    for keyword in keywords:
        query = []
        for i in range(len(grounding)):
            query.append(keyword)
        template[keyword] = generate(model=t5_model, tokenizer=t5_tokenizer, grounding=grounding, input=query, output=answers, case_num=200, no_extra=True, device=device)

        prompt_dict[keyword] = {}
        for t in tqdm(template[keyword]):
            seq =[]
            for i in range(len(grounding)):
                if len(t)==2:
                    tmp_prompt = (t[0].replace('?','') + ' '+ keyword + ' '+ t[1]).replace('.','') # .replace('?','')
                    seq.append( grounding[i] + ' '+ tmp_prompt + ' ' + answers[i].lower())
                elif len(t)==3:
                    tmp_prompt = (t[1].replace('?','') + ' '+ keyword + ' '+ t[2]).replace('.','') # .replace('?','')
                    seq.append(t[0].replace('.','').replace('?','') + ' '+ grounding[i] + ' '+ tmp_prompt + ' ' + answers[i].lower())
            score = get_ppl(gpt_model,seq,gpt_tokenizer,eval_num=384,device=device)
            if len(t)==2:
                prompt_dict[keyword][score] = tmp_prompt
                #new_prompts[score] = tmp_prompt
            elif len(t)==3:
                prompt_dict[keyword][score] = (t[0].replace('.','').replace('?',''), tmp_prompt)
                #if isinstance(t, list):
                #    if len(t)==2:
                #        seq.append( grounding[i] + ' '+t[0] + ' '+ keyword + ' '+ t[1])
                #    elif len(t)==3:
                #        seq.append(t[0] + ' '+ grounding[i] + ' '+t[1] + ' '+ keyword + ' '+ t[2])
                    #seq.append( t[0] + ' '+ query[i] + ' '+ t[1])
            #scores[keyword].append(get_ppl(model=model, tokenizer=tokenizer, seq=seq, eval_num=4096, batch_size=32)) # evaluate
        #origin_scores = copy.deepcopy(scores[keyword])
        #scores[keyword].sort(reverse=True)
        #for score in scores[keyword]:
        #    prompt = template[keyword][origin_scores.index(score)]
        #    if keyword not in prompt_dict:
        #        prompt_dict[keyword] = {}
        #    prompt_dict[keyword][score] =prompt
    #json.dump(prompt_dict,open(('data/prompts/intent.json'), 'w'), indent=2)
    json.dump(prompt_dict,open(('data/new_prompts/' + domain + '_origin.json'), 'w'), indent=2)
    return

def generate_template_instruction():
    keywords = json.load(open('natural-instructions-2.7/keywords_final.json', 'r'))
    instances = json.load(open('natural-instructions-2.7/task_instances.json', 'r'))
    device = 6
    torch.cuda.set_device(device)
    gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')
    gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt_model = gpt_model.cuda()
    gpt_model.eval()
    t5_model = T5ForConditionalGeneration.from_pretrained('t5-large')
    t5_tokenizer = T5Tokenizer.from_pretrained('t5-large')
    # for generation chinese prompts
    #from transformers import T5ForConditionalGeneration, BertTokenizer
    #tokenizer=BertTokenizer.from_pretrained('IDEA-CCNL/Randeng-T5-Char-700M-Chinese', add_special_tokens=False)
    #model=T5ForConditionalGeneration.from_pretrained('IDEA-CCNL/Randeng-T5-Char-700M-Chinese')

    t5_model = t5_model.cuda()
    t5_model.eval()
    prompt_dict = {}
    template = {}
    #scores = {}
    seq = []
    for task, ks in tqdm(keywords.items()):
        prompt_dict[task] = {}
        template[task] = {}
        cases = instances[task]
        t_keywords = [task.lower()]
        t_keywords.extend(task.lower().split())
        for k in ks:
            if ' '.join(k) not in t_keywords:
                t_keywords.append(' '.join(k))
        
        for keyword in t_keywords:
            query = []
            grounding = []
            answers = []
            for i in range(len(cases)):
                grounding.append(cases[i]['input'])
                query.append(keyword)
                answers.append(cases[i]['output'][0])
            template[task][keyword] = generate(model=t5_model, tokenizer=t5_tokenizer, grounding=grounding, input=query, output=answers, case_num=60, no_extra=True, device=device)

            prompt_dict[task][keyword] = {}
            for t in tqdm(template[task][keyword]):
                seq =[]
                for i in range(len(grounding)):
                    if len(t)==2:
                        tmp_prompt = (t[0].replace('?','') + ' '+ keyword + ' '+ t[1]).replace('.','') # .replace('?','')
                        seq.append( grounding[i] + ' '+ tmp_prompt + ' ' + answers[i].lower())
                    elif len(t)==3:
                        tmp_prompt = (t[1].replace('?','') + ' '+ keyword + ' '+ t[2]).replace('.','') # .replace('?','')
                        seq.append(t[0].replace('.','').replace('?','') + ' '+ grounding[i] + ' '+ tmp_prompt + ' ' + answers[i].lower())
                score = get_ppl(gpt_model, seq, gpt_tokenizer, eval_num=196 if len(seq)>196 else len(seq), device=device)
                if len(t)==2:
                    prompt_dict[task][keyword][score] = tmp_prompt
                    #new_prompts[score] = tmp_prompt
                elif len(t)==3:
                    prompt_dict[task][keyword][score] = (t[0].replace('.','').replace('?',''), tmp_prompt)

    json.dump(prompt_dict,open(('data/new_prompts/instuction_prompts_origin_3.json'), 'w'), indent=2)
    return

def evaluate_ppl():
    device=1
    torch.cuda.set_device(device)
    eval_num = 8192
    grounding, answers = get_state() # get_intent()
    all_prompts = []
    #prompts = json.load(open('data/prompts/intent.json','r'))
    prompts = json.load(open('data/prompts/state.json','r'))
    for k,v in prompts.items():
        if k!='base':
            for s,p in v.items():
                all_prompts.append(p)
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = model.cuda()
    model.eval()
    case=[]
    new_prompts={}
    for k,v in prompts.items():
        if k=='user state' :  #k=='user intent' : # k!='base'
            for s,p in v.items():
                case = []
                for num in range(eval_num):
                    if len(p)==2:
                        tmp_prompt = (p[0].replace('?','').replace('!','') + ' '+ k + ' '+ p[1]).replace('.','') # .replace('?','')
                        case.append( grounding[num] + ' '+ tmp_prompt + ' ' + answers[num].lower())
                    elif len(p)==3:
                        tmp_prompt = (p[1].replace('?','').replace('!','') + ' '+ k + ' '+ p[2]).replace('.','') # .replace('?','')
                        case.append(p[0].replace('.','').replace('?','') + ' '+ grounding[num] + ' '+ tmp_prompt + ' ' + answers[num].lower())
                score = get_ppl(model,case,tokenizer,eval_num,device=device)
                if len(p)==2:
                    new_prompts[score] = tmp_prompt
                elif len(p)==3:
                    new_prompts[score] = [p[0].replace('.','').replace('?',''), tmp_prompt]
    #json.dump(new_prompts,open(('data/prompts/intent_new.json'), 'w'), indent=2)  
    json.dump(new_prompts,open(('data/prompts/' + '_new.json'), 'w'), indent=2)    
        #else:
        #    new_prompts[v] = k

def get_ppl(model, seq, tokenizer, eval_num=4096, batch_size=16,device=0):
    score = []
    acc = 0
    metrics = 0
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #tokenizer.pad_token = tokenizer.eos_token
    #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    random.shuffle(seq)
    if len(seq)<eval_num:
        eval_num = len(seq)
    with torch.no_grad():
        for i in range(int(eval_num/batch_size)):
            tokenized = tokenizer(seq[batch_size*i:batch_size*(i+1)], return_tensors="pt", padding=True, max_length=512,truncation = True) # truncation = True 
            tokenized = tokenized.to(device)
            outputs=model(**tokenized, labels=tokenized['input_ids'])
            loss=outputs.loss
            loss1 = loss.cpu().tolist()
            if loss1<15:
                logit = loss1
                ppl=math.exp(loss1)
            else:
                ppl=math.exp(15.0)
            score.append(ppl)
                
    return sum(score)/eval_num

def analysis(domain):
    result = json.load(open('data/new_prompts/' + domain + '_origin.json','r'))
    final_score = [] 
    num = 0
    for k,states in result.items():
        scores = []
        if k!='base' : # check if the quality is okay in ''
            for s, _ in states.items(): # intents
                scores.append(float(s))
            print(k)
            print(sum(scores)/len(scores))
            final_score.append(sum(scores)/len(scores))
            num = num + len(scores)
        else:
            print('base')
            print(states)
    print('avg:')
    print(sum(final_score)/len(final_score))
    print(num)
    all_prompts = 0
    for domain in ALL_DOMAINS:
        file = 'data/new_prompts/' + domain + '_final.json'
        prompt_num = len(json.load(open(file,'r')))
        print(f"{domain}:{prompt_num}")
        all_prompts = all_prompts + prompt_num
    print(all_prompts)
    return

def get_final(domain):
    #intents = json.load(open('data/prompts/intent_new.json','r'))
    #result = json.load(open('data/new_prompts/state_gen.json','r'))
    result = json.load(open('data/new_prompts/' + domain + '_origin.json','r'))

    ignore_words = {'emotion':['positive','negative','neutral','no emotion'],'summary':[],'intent':[],'answer':['1','2','3']
    , 'response':[], 'generated':[], 'choice':['1','2','3'], 'key information':['*','>','1','2','3'], 'summary_zh':['【'],
    'nlu_zh':['示 意图'], 'sql_zh':[], 'similarity_zh':[], 'emotion_zh':[], 'act_zh':[], 'domain_zh':[]}
    scores = []
    prompts = []
    prompts_simplified = []
    prompts_d = [] # only used in dialoglue-related

    base_score = result['base']
    for k,states in result.items():
        scores = []
        if k!='base' and k!='': # check if the quality is okay in ''
            for s, _ in states.items(): # intents
                scores.append(float(s))
            scores.sort()
            for s in scores[:30]:
                if s < base_score:
                    p = states[str(s)]
                    if p not in prompts:
                        flag = 0
                        if isinstance (p,list):
                            for ignore in ignore_words[domain]:
                                if ignore in ''.join(p).lower():
                                    flag = 1
                            if flag == 0:
                                prompts.append(p)
                                if p[1] not in prompts_simplified:
                                    prompts_simplified.append(p[1])
                                    prompts_d.append((p[0] + ' user: ', p[1]))
                        else:
                            for ignore in ignore_words[domain]:
                                if ignore in p:
                                    flag = 1
                            if flag == 0:
                                prompts.append(p)
                                if p not in prompts_simplified:
                                    prompts_simplified.append(p)
                                    prompts_d.append((p[0] + ' user: ', p))
    #json.dump(prompts_d, open('data/new_prompts/state_dialoglue.json','w'), indent=2)
    #json.dump(prompts, open('data/new_prompts/state.json','w'), indent=2)
    json.dump(prompts, open('data/new_prompts/' + domain + '_selected.json','w'), indent=2, ensure_ascii=False)
    json.dump(prompts_simplified, open('data/new_prompts/' + domain + 'simplified_selected.json','w'), indent=2)
    if domain in ['intent', 'state']:
        json.dump(prompts_d, open('data/new_prompts/' + domain + 'simplified_selected_dialoglue.json','w'), indent=2)
    """
    for s, _ in intents.items(): # intents
        scores.append(float(s))
    scores.sort()
    for s in scores[:10]:
        p = intents[str(s)]
        if isinstance (p,list):
            prompts_d.append((p[0] + ' user: ', p[1]))
            prompts.append(p)
        else:
            prompts_d.append(('user: ', p))
            prompts.append(p)
    json.dump(prompts_d, open('data/new_prompts/intent_dialoglue.json','w'), indent=2)
    json.dump(prompts, open('data/new_prompts/intent.json','w'), indent=2)
    """

def collect_task_and_instructions():
    wanted_cats = ['Fill in The Blank', 'Question Understanding', 'Question Answering', 'Named Entity Recognition',
    'Text Categorization', 'Commonsense Classification', 'Dialogue Generation', 'Data to Text', 'Summarization',
    'Keyword Tagging', 'Dialogue State Tracking']
    data_dir = 'natural-instructions-2.7/tasks'
    for cat in wanted_cats:
        cat_data = []
        for file in os.listdir(data_dir):
            if file!='README.md':
                dataset = json.load(open(os.path.join(data_dir,file),'r'))
                if (dataset["Input_language"] == ["English"]) and (dataset["Output_language"] == ["English"]) and(
                    dataset["Instruction_language"] == ["English"]) and (dataset['Categories'][0]==cat) :
                    cat_data.append (dataset)
        json.dump(cat_data, open('data/instructions/' + cat +'.json','w'), indent=2)          
    return

def generate_keyword_using_instructions(cat='Fill in The Blank'):
    dir = 'data/instructions/' + cat +'.json'
    datas = json.load(open(dir,'r'))
    instructions = []
    for data in datas:
        instructions.append(data['Definition'])
    return
def process_template_instruction():
    """
    tmp = json.load(open('natural-instructions-2.7/keywords_final.json','r'))
    keywords = ['']
    for task, ks in tqdm(tmp.items()):
        keywords.extend(task.lower().split())
        if task.lower() not in keywords:
            keywords.append(task.lower())
        for k in ks:
            if ' '.join(k) not in keywords:
                keywords.append(' '.join(k))
    """
    total = 0
    scores = {}
    prompts = json.load(open('data/new_prompts/instuction_prompts_origin_3.json','r'))
    for task, kps in prompts.items():
        scores[task] = []
        for keyword, prompts in kps.items():
            for score,p in prompts.items():
                scores[task].append(float(score))
    for task, ss in scores.items():
        avg = sum(ss)/len(ss)
        total += sum(s<0.7*avg for s in ss)
    return

def rescore(domain):
    prompts = json.load(open('data/new_prompts/' + domain + '_selected.json','r'))
    get_domain_data = {'intent':get_intent, 'state':get_state,
    'emotion':get_emotion, 'summary':get_summary, 'answer':get_qa, 'generated':get_nlg
    , 'response':get_response, 'choice':get_choice, 'key information':get_txt2sql}
    grounding, answers = get_domain_data[domain]()
    device=0
    torch.cuda.set_device(device)
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = model.cuda()
    model.eval()
    prompt_dict = {}
    for t in tqdm(prompts):
        seq =[]
        for i in range(len(grounding)):
            if isinstance(t,str):
                seq.append( grounding[i] + ' '+ t+ ' ' + answers[i].lower())
            elif len(t)==2:
                seq.append(t[0] + ' '+ grounding[i] + ' '+ t[1] + ' ' + answers[i].lower())
        score = get_ppl(model,seq,tokenizer,eval_num=1024,device=device)
        prompt_dict[score] = t
    #prompt_dict =  json.load(open('data/new_prompts/' + domain + '_rescored.json','r'))
    new_dict = {}
    scores = []
    for k in prompt_dict:
        score = float(k)*(len(''.join(prompt_dict[k]))+10)
        scores.append(score)
        new_dict[score] = prompt_dict[k]
    scores.sort()
    wanted = scores[:60]
    wanted_prompts = []
    for score in wanted:
        wanted_prompts.append(new_dict[score]) # score

    json.dump(wanted_prompts, open('data/new_prompts/' + domain +'_rescored.json','w'), indent=2, ensure_ascii=False)
    #json.dump(wanted_prompts, open('data/new_prompts/' + domain +'_rescored.json','w'), indent=2) # for zh    
    return

def post_process():
    final = []
    # sql_zh_selected
    # similarity_zh_selected
    dat = json.load(open('data/new_prompts/act_zh_selected.json', 'r'))
    for i in dat:
        if isinstance(i, list):
            final.append(i[1].replace(' ', '').replace('。', '').replace('英语', '').replace('，', '').replace('~', ''))
        else:
            final.append(i.replace(' ', '').replace('。', '').replace('英语', '').replace('，', '').replace('~', ''))
    final = list(set(final))
    # similarity_zh_prompts.json
    json.dump(final, open('data/new_prompts/act_zh_prompts.json', 'w'), indent=2, ensure_ascii=False)
    return

def scripts():
    dat = json.load(open('data/new_prompts/summary_zh_prompts.json', 'r'))
    dat.extend(json.load(open('data/new_prompts/summary_zh_prompts1.json', 'r')))
    data = []
    for d in dat:
        if '【' not in d:
            data.append(d.replace('?','').replace(':','：').replace('：：','：').replace('！','').replace('。',''))
    final = list(set(data))
    json.dump(final, open('data/new_prompts/summary_zh_prompts_new.json', 'w'), indent=2, ensure_ascii=False)
    return
    
if __name__ == '__main__':
    #domain in ['intent', 'state', 'emotion', 'summary', 'answer', 'generated'
    #, 'response', 'choice', 'key information', 'summary_zh', 'nlu_zh',
    # 'sql_zh', 'act_zh', 'dst_zh', 'domain_zh', 'emotion_zh', 
    # 'similarity_zh']
    domain = 'domain_zh' # dialog summary in chinese
    #main()
    #collect_prompt_result()
    #generate_template_instruction()
    #process_template_instruction()
    #generate_template_using_keyword(domain)
    #evaluate_ppl()
    #analysis(domain)
    #get_final(domain)
    #rescore(domain)
    #collect_task_and_instructions()
    #generate_keyword_using_instructions()
    post_process()
    #scripts()