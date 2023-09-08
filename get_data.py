from fileinput import filename
import json
import os
from pydoc import doc
import re
import pandas as pd
import numpy as np
import random
import queue
import datasets
import copy

#from space.data.dataset import LazyDataset
#import parlai.core.build_data as build_data
#from parlai.tasks import multiwoz_v20 as mul
#build_data(mul)
#tmp = mul.build()

from datasets import load_dataset
from promptsource.templates import DatasetTemplates
#maximedb/wow

intent_prompts = json.load(open('data/new_prompts/intent_final.json','r'))
state_prompts = json.load(open('data/new_prompts/state_final.json','r'))

#wizard of internet 8614 dials
def add_endmark(sentence):
    question = ['how', 'do you', 'are there', 'what', 'can you', 'is there', 'why', 'where']
    if sentence != '':
        if sentence[-1] not in ['.','?','!'] and (sentence[-2] if len(sentence)>1 else sentence[-1]) not in ['.','?','!']:
            flag = 0
            for q in question:
                if q in sentence:
                    flag=1
                    break
            if flag==0:
                sentence = sentence + '.'
            else:
                sentence = sentence + '?'
    return(sentence)

def get_context(context):
    TURN_NUM = 6
    new_context = ''
    #if len(context)>TURN_NUM:
    for turn in context[-TURN_NUM:]:
        new_context = new_context + ' ' + turn['role'] + ': ' + add_endmark(turn['text'])
    #else:
    #    for turn in context:
    #        new_context = new_context + ' ' + turn['role'] + ': ' +turn['text']
    return new_context

def get_woi(task=0):#get wizard of internet, to do: add search no-search
    tasks = ['search','response generation']
    description = 'Task description: given the context and grounding, search the web or predict the next utterance.'
    file = '/home/caiyc/miniconda3/envs/prompt/lib/python3.7/site-packages/data/wizard_of_interent/train.jsonl'
    query = []
    answer = []
    grounding = []
    template_input = {}
    data = []  #:list of cases, data[0]['3842']['apprentice_persona']# remove '\n'  #['dialog_history]:list of turns: 'action','text', 'context'
    tmp = '1'
    with open(file, "r", encoding="utf-8") as fp:  #8614 dialog,167667turns
        while tmp != "":
            tmp = fp.readline()
            if tmp != '':
                data.append(json.loads(tmp))
    for case in data:
        for id, log in case.items():
            dialog = log['dialog_history']
            context = 'Apprentice persona: '+ log['apprentice_persona'].replace('\n',' ')
            background = {}
            for turn in dialog:
                if turn['action']=='Wizard => SearchAgent':
                    if task == 0:
                        grounding.append(context)
                        query.append('What information should you search?')
                        answer.append(turn['text'])
                elif turn['action']=='Wizard => Apprentice':
                    background_text = ''
                    for num in range(len(turn['context']['selected_contents'])-1):
                        for piece_num in range(len(turn['context']['selected_contents'][num+1])):
                            if turn['context']['selected_contents'][num+1][piece_num] and len(background['contents'][num]['content'])>piece_num :
                                background_text = background_text + ' ' + background['contents'][num]['content'][piece_num]
                    if task == 1:
                        grounding.append(background_text)
                        query.append(context + ' ' + turn['action'].split('=>')[0] + ': ')
                        answer.append(turn['text'])
                    context = context + ' ' + turn['action'].split('=>')[0] +': '+ turn['text']
                elif turn['action']=='SearchAgent => Wizard':
                    background = turn['context']
                else:
                    context = context + ' ' + turn['action'].split('=>')[0] +': '+ turn['text']
    return grounding, query, answer, template_input, description

####
#docqa
#squad
def get_squad(): # with_grounding=False # t0 template not added yet
    #qas =0
    task = 'question answering'
    description = 'Task description: given the passage, answer the question.'
    prompts = DatasetTemplates('squad') # '_'
    dataset = load_dataset('squad')#, split="train"
    train = dataset['train']
    qs = train['question']
    ans = train['answers']
    grounds = train['context']
    template_input = {}
    #file = '/home/caiyc/miniconda3/envs/prompt/lib/python3.7/site-packages/data/SQuAD/train-v1.1.json'
    #dataset = json.loads(open(file).read())['data']  #442 passages 'title','paragraphs':list of 'context','qas':['answer']['text']
    query = []
    answer = []
    grounding = []
    #for para in dataset:
    #for passage in para['paragraphs']:
    for num in range(len(qs)):
        """
        for t in prompts.all_template_names:
            if t not in template_input:
                template_input[t] = []
            prompt = prompts[t]
            template_input[t].append(prompt.apply(train[num]))
        """
        context = grounds[num]
        queries = qs[num]
        answers = ans[num]['text']
        # for num1 in range(len(queries)):
        query.append(queries)
        if answers[0]=='CANNOTANSWER': # can be used for query task
            answer.append('can not answer.')
        else:
            answer.append(answers[0])
        grounding.append(context)
    return grounding, query, answer, template_input, description
    
#quac
def get_quac():
    """
    file = '/home/caiyc/miniconda3/envs/prompt/lib/python3.7/site-packages/data/QuAC/train.txt'
    data = []
    tmp = '1'
    with open(file, "r", encoding="utf-8") as fp:
        while tmp != "":
            tmp = fp.readline()
            if tmp != '':
                tmp_list = tmp.split('\t')
                data.append(tmp)
    """
    path = 'data/pre_prompt/quac.json'
    task = 'question answering'
    description = 'Task description: given the passage, answer the question.'
    if os.path.exists(path):
        result = json.load(open(path, 'r'))
        grounding = result['grounding']
        query = result['query']
        answer= result['answer']
        template_input = result['template_input']
    else:
        query = []
        answer = []
        grounding = []
        prompts = DatasetTemplates('quac')
        #template = prompts.all_template_names
        #template.append(description)
        dataset = load_dataset('quac')#, split="train", no en-fr available
        template_input = {}
        train = dataset['train']
        des_hug = train.description.replace('\n', ' ')
        context = train['context']
        question = train['questions']
        answers = train['answers']
        for num in range(len(question)):
            for t in prompts.all_template_names:
                if t not in template_input:
                    template_input[t] = []
                prompt = prompts[t]
                template_input[t].append(prompt.apply(train[num]))
            con = context[num]
            qs = question[num]
            ans = answers[num]['texts']
            for num1 in range(len(ans)):
                an = ans[num1][0]
                if an != 'CANNOTANSWER':
                    answer.append(an)
                else:
                    answer.append('Cannot answer.')
                query.append(qs[num1])
                grounding.append(con.replace('CANNOTANSWER', ''))
        result = {'grounding':grounding,'query':query,'answer':answer,'template_input':template_input}
        json.dump(result, open(path, 'w'),indent=2)
    return grounding, query, answer, template_input, description
#with open(file, "r", encoding="utf-8") as fp:
#    data = fp.read()
#dataset = json.loads(open(file).read())

def get_doqa(): #no prompt
    files = ['extra_data/doqa/doqa_dataset/doqa-cooking-train-v2.1.json','extra_data/doqa/doqa_dataset/doqa-cooking-dev-v2.1.json','extra_data/doqa/doqa_dataset/doqa-cooking-test-v2.1.json','extra_data/doqa/doqa_dataset/doqa-movies-test-v2.1.json','extra_data/doqa/doqa_dataset/doqa-travel-test-v2.1.json'] #ir_scenario
    data = []
    task = 'question answering'
    template_input = {}
    for file in files:
        data.extend(json.loads(open(file).read())['data'])
    query = []
    answer = []
    grounding = []
    description = 'Task description: given the background and context, answer the question.'
    for dial in data: #1037
        background = dial['background']
        main = dial['paragraphs'][0]
        context = ''
        qas = main['qas']
        for qa in qas:
            grounding.append(background)
            context = context + ' user: ' +  qa['question']
            query.append(context + ' system: ')
            answer.append(qa['answers'][0]['text'].lower())
            context = context + ' system: ' + qa['answers'][0]['text']
    return grounding, query, answer,template_input, description

##dataset = load_dataset('coqa')
#https://raw.githubusercontent.com/huggingface/datasets/2.4.0/datasets/coqa/coqa.py

#cmu_dog
#/home/caiyc/miniconda3/envs/prompt/lib/python3.7/site-packages/data

def get_narrativeqa():
    description = 'Task description: given the passage, answer the question.'
    path = 'data/pre_prompt/narrativeqa.json'
    if os.path.exists(path):
        result = json.load(open(path, 'r'))
        grounding = result['grounding']
        query = result['query']
        answers= result['answer']
        template_input = result['template_input']
    else:
        answers = []
        grounding = []
        query = []
        prompts = DatasetTemplates('narrativeqa')
        all_prompts = prompts.templates
        dataset = load_dataset('narrativeqa')
        template_input = {}
        train = dataset['train']
        #des_hug = train.description.replace('\n', ' ')
        document = train['document']
        question = train['question']
        answer = train['answers']
        for num in range(len(question)):
            for t in prompts.all_template_names:
                if t not in template_input:
                    template_input[t] = []
                prompt = prompts[t]
                template_input[t].append(prompt.apply(train[num]))
            con = document[num]['summary']['text']
            grounding.append(con)
            query.append(question[num]['text'])
            answers.append(answer[num][0]['text'])
        result = {'grounding':grounding,'query':query,'answer':answers,'template_input':template_input}
        json.dump(result, open(path, 'w'),indent=2)
    return grounding, query, answers, template_input, description
                   
def get_race(): # no template
    path = 'data/pre_prompt/race.json'
    description = 'Task description: given the passage, answer the question.'
    if os.path.exists(path):
        result = json.load(open(path, 'r'))
        grounding = result['grounding']
        question = result['query']
        answers= result['answer']
        template_input = result['template_input']
    else:
        answers = []
        prompts = DatasetTemplates('race')
        all_prompts = prompts.templates
        dataset = load_dataset("race",'all')
        template_input = {}
        train = dataset['train']
        grounding = train['article']
        question = train['question']
        option = train['options']
        answer = train['answer']
        choice2answer = {'A':0,'B':1,'C':2,'D':3}
        for num in range(len(question)):
            for t in prompts.all_template_names:
                if t not in template_input:
                    template_input[t] = []
                prompt = prompts[t]
                template_input[t].append(prompt.apply(train[num]))
            choice = choice2answer[answer[num]]
            answers.append(option[num][choice])
        result = {'grounding':grounding,'query':question,'answer':answers,'template_input':template_input}
        json.dump(result, open(path, 'w'),indent=2)
    return grounding, question, answers, template_input,description
####
#DIALQA
def get_ddrel(): #no prompt
    task = 'question answering' # speaker detection
    file = 'extra_data/ddrel/train.txt'
    description ='Task description: given the dialog, find the relation bewtween the two speakers.'
    tmp = '1'
    grounding = []
    question = []
    answers = [] 
    template_input = {}
    label_relation = json.load(open('extra_data/ddrel/relation_label.json','r'))
    with open(file, "r", encoding="utf-8") as fp:
        while tmp != "":
            tmp = fp.readline()
            if tmp != '':
                t = json.loads(tmp)
                dial = ''
                for turn in t['context']:
                    if dial =='':
                        dial = turn
                    else:
                        dial = dial + ' ' + turn
                grounding.append(dial)
                question.append('What is the relation between two speakers? ') # can be change to key words
                answers.append(label_relation[int(t['label'])-1])
    return grounding, question, answers, template_input, description

def get_friends_qa(): # no prompt
    file = 'extra_data/FriendsQA/dat/friendsqa_trn.json' #973 docs, ut:21607, qa:9874
    task = 'question answering'
    description ='Task description: given the dialog, answer the question.'
    grounding = []
    question = []
    answers = [] 
    template_input = {}
    dataset = json.loads(open(file).read())
    for dial in dataset['data']:
        dialog = dial['paragraphs'][0]['utterances:']
        t_dial = ''
        for turn in dialog:
            t_dial = t_dial + ' ' + turn['speakers'][0] + ': ' + (turn['utterance'] if isinstance(turn['utterance'],str) else turn['utterance'][0])
        qas = dial['paragraphs'][0]['qas']
        for qa in qas:
            question.append(qa['question'])
            answers.append(qa['answers'][0]['answer_text'])
            grounding.append(t_dial)
        
    return grounding, question, answers, template_input, description

def get_molweni(): # no prompt
    file = 'extra_data/Molweni/MRC/train.json' #8771 dials, 77374	24,682
    dpfile = 'extra_data/Molweni/DP/train.json'
    dataset = json.loads(open(file).read())
    description ='Task description: given the dialog, answer the question.'
    grounding = []
    question = []
    answers = [] 
    template_input = {}
    for dial in dataset['data']['dialogues']:
        dialog = dial['edus'] #turn = 'text'
        t_dial = ''
        for turn in dialog:
            t_dial = t_dial + ' ' + turn['speaker'] + ': ' + turn['text']
        qas = dial['qas']
        for qa in qas:
            if qa['answers']!=[]:
                question.append(qa['question'])
                answers.append(qa['answers'][0]['text'])
                grounding.append(t_dial)
    return grounding, question, answers, template_input, description

def get_mutual(): # no prompt
    dataset = []
    from extra_data.MuTual.baseline.utils_multiple_choice import MuTualProcessor
    description ='Task description: given the dialog context, predict the next utterance.'
    grounding = []
    question = []
    answers = [] 
    template_input = {}
    dir = 'extra_data/MuTual/data/mutual' #7088 dials
    processor = MuTualProcessor()
    train_data = processor.get_train_examples(dir)
    for item in train_data:
        context = item.contexts[0]
        label = item.label
        correct_answer = item.endings[int(label)]
        question.append((context+ correct_answer[:3]).replace('m :','user:').replace('f :','system:'))
        answers.append(correct_answer[3:])
    return grounding, question, answers, template_input, description

def get_dialog_re(): # 
    path = 'data/pre_prompt/dialog_re.json'
    description = 'Task description: given the dialog, find the relation bewtween the speakers.'
    if os.path.exists(path):
        result = json.load(open(path, 'r'))
        grounding = result['grounding']
        question = result['query']
        answers= result['answer']
        template_input = result['template_input']
    else:
        template_input = {}
        file = 'extra_data/dialogre/data/train.json' #emo and act trained in daily dialog
        dataset = json.loads(open(file).read())  #1073 sessions, 14024 turns
        prompts = DatasetTemplates('dialog_re')
        dataset = load_dataset('dialog_re')
        train = dataset['train']
        dialog = train['dialog']
        question = []
        answers = []
        grounding = []
        relations = train['relation_data']
        for num in range(len(dialog)):
            d = dialog[num]
            dial =''
            for turn in d:
                if dial != '':
                    dial = dial + ' ' + turn
                else:
                    dial = turn
            r = relations[num]
            for num1 in range(len(r['x'])):
                grounding.append(dial)
                question.append(('What is the relation between ' + r['x'][num1] + ' and ' + r['y'][num1] + '?'))
                if ':'in r['r'][num1][0]:
                    answers.append(r['r'][num1][0].split(':')[1])
                else:
                    answers.append(r['r'][num1][0])
        result = {'grounding':grounding,'query':question,'answer':answers,'template_input':template_input}
        json.dump(result, open(path, 'w'),indent=2)
    return grounding, question, answers, template_input,description

def get_dream():
    path = 'data/pre_prompt/dream.json'
    description = 'Task description: given the passage, answer the question.'
    if os.path.exists(path):
        result = json.load(open(path, 'r'))
        grounding = result['grounding']
        question = result['query']
        answer= result['answer']
        template_input = result['template_input']
    else:
        query = []
        answer = []
        grounding = []
        prompts = DatasetTemplates('dream')
        dataset = load_dataset('dream')
        template_input = {}
        train = dataset['train']
        #des_hug = train.description.replace('\n', ' ')
        dialog = train['dialogue'] #: serveral turns of lists: W,M
        answer = train['answer'] # a choice from 
        question = train['question'] # 4 each article
        for num in range(len(question)):
            for t in prompts.all_template_names:
                if t not in template_input:
                    template_input[t] = []
                prompt = prompts[t]
                template_input[t].append(prompt.apply(train[num]))
            context = ''
            for turn in dialog[num]:
                context = context +' '+ turn
            grounding.append(context)
        result = {'grounding':grounding,'query':question,'answer':answer,'template_input':template_input}
        json.dump(result, open(path, 'w'),indent=2)
    return grounding, question, answer, template_input, description

####
#chat
def get_dailydialog(task = 0, FSB=False): # can train classification model
    dataset = load_dataset("daily_dialog")#11118 dialogs success
    train = dataset['train']
    description = 'Task description: given the utterance, find the intention or the emotion.'
    template_input = {}
    #prompts = DatasetTemplates('daily_dialog')
    question = []
    answer = []
    grounding = []
    FSB_result = []
    dialog = train['dialog'] #list of turns
    act_list = ['inform','question','directive','commissive']
    act = train['act'] # 1-4: ['inform','question','directive','commissive']
    emo_list = ['no emotion', 'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
    emo = train['emotion'] #0-6: ['no emotion', 'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
    for num in range(len(dialog)):
        context = ''
        dial = {"meta":[],"dialogue":[]}
        for turn_num in range(len((dialog[num]))):
            if turn_num % 2 == 0:
                temp = [dialog[num][turn_num]]
                temp_meta = [{"emotion":emo_list[emo[num][turn_num]], "act": act_list[act[num][turn_num]-1]}]
            else:
                temp.append(dialog[num][turn_num])
                temp_meta.append({"emotion":emo_list[emo[num][turn_num]], "act": act_list[act[num][turn_num]-1]})
                dial["dialogue"].append(temp)
                dial["meta"].append(temp_meta)
            """
            for ids_t, turn in enumerate(obj["dialogue"]):
                if ids_t % 2 == 0:
                    temp = [turn["text"]]
                    temp_meta = [{"emotion":turn["emotion"], "act": turn["act"]}]
                else:
                    temp.append(turn["text"])
                    temp_meta.append({"emotion":turn["emotion"], "act": turn["act"]})
                    dial["dialogue"].append(temp)
                    dial["meta"].append(temp_meta)

            if len(temp) == 1:
                temp.append("")
                dial["dialogue"].append(temp)
                dial["meta"].append({})

            assert len(dial["meta"]) == len(dial["dialogue"])
            data.append(dial)
            """
            if task == 0:
                input = random.sample(intent_prompts,k=1)[0]
                if isinstance(input,list):
                    question.append(input[1].replace('user','speaker').replace('User','Speaker'))
                    grounding.append(dialog[num][turn_num]) # input[0] + ' '+
                    answer.append(act_list[act[num][turn_num]-1]) 
                else:
                    question.append(input.replace('user','speaker').replace('User','Speaker'))
                    grounding.append(dialog[num][turn_num]) # input[0] + ' '+
                    answer.append(act_list[act[num][turn_num]-1]) 
            elif task == 1:
                question.append(dialog[num][turn_num]) #+ ' The emotion is: ')
                answer.append(emo_list[emo[num][turn_num]])
        if len(temp) == 1:
            temp.append("")
            dial["dialogue"].append(temp)
            dial["meta"].append({})
        FSB_result.append(dial)
    if FSB:
        json.dump(FSB_result, open('data/extra/dailydialog.json','w'))
    return grounding, question, answer, template_input, description

def get_personachat(FSB=False): # no-supervision, role
    dataset = load_dataset("bavard/personachat_truecased", split="train")# 17877sessions with 131438 turns .lower() success
    description = 'Task description: given the personality and context, predict the next utterance.'
    template_input = {}
    question = []
    answer = []
    grounding = []
    per = dataset['personality'] #list of personalities
    FSB_result = []
    FSB_tmp = {}
    #dataset['candidates'] list of 20 utts, not used
    hist = dataset['history'] # list of turns
    answer = dataset['candidates']
    for num in range(len(per)):
        personality = ''
        for p in per[num]:
            personality = personality + ' ' + p
        if len(hist[num]) == 1: # new_dialog
            context = 'user' + ': ' + hist[num][0]
            if FSB_tmp!={}:
                FSB_result.append(FSB_tmp)
            FSB_tmp = {'meta': per[num], 'dialogue':[(hist[num][-1].lower(),answer[num][-1].lower())]}
        else:
            grounding.append(personality)
            question.append(context + ' system ' + ': ')
            answer.append(hist[num][-2])
            context = context + ' system: ' + hist[num][-2]
            grounding.append(personality)
            question.append(context + ' user: ' )
            answer.append(hist[num][-1])
            context = context + ' user' + ': ' + hist[num][-1]
            FSB_tmp['dialogue'].append((hist[num][-1].lower(),answer[num][-1].lower()))
    if FSB:
        json.dump(FSB_result, open('data/extra/persona.json','w'))
    return grounding, question, answer, template_input, description

    #dataset['utterance_idx']:turn_num

def get_metawoz(): # no-supervision, system-training
    description = 'Task description: given the domain and context, predict the next utterance.'
    template_input = {}
    #prompts = DatasetTemplates('meta_woz')
    question = []
    answer = []
    grounding = []
    dataset = load_dataset("meta_woz", split="train")  #success 37884 session
    #dataset['id'] 
    #dataset['user_id'] #list of 20 utts, not used
    bot_id = dataset['bot_id'] #list of turns
    domain = dataset['domain']
    #dataset['task_id']
    dial = dataset['turns'] #list of turns,0 user, 1 bot
    role = ['user','system']
    for num in range(len(domain)):
        d = domain[num].lower().split('_')[0]
        context = ''
        for turn_num in range(len(dial[num])):
            if turn_num>1 and (turn_num%2)==0:
                grounding.append('domain: ' + d)
                question.append(context + ' ' + role[turn_num%2]+ ': ')
                answer.append(dial[num][turn_num])
            context = context +' ' +role[turn_num%2] + ': ' + dial[num][turn_num]

    return grounding, question, answer, template_input, description

def get_empathetic_dialogues():
    # the sentiment is somewhat like domain in this dataset
    description = 'Task description: given the sentiment and context, predict the next utterance.'
    template_input = {}
    question = []
    answer = []
    grounding = []
    dataset = load_dataset("empathetic_dialogues", split="train") 
    id = dataset['conv_id'] #list of personalities
    emos = dataset['context'] # emotion each turn
    #dataset['prompt']: case?
    #dataset['speaker_idx']:
    dialog = dataset['utterance']
    tmp_id = ''
    turn = 0
    role = ['user','system']
    context = ''
    for num in range(len(emos)):
        if id[num]!=tmp_id: # new dial
            tmp_id = id[num]
            context = 'user: ' + dialog[num]
            turn = 0
        else:
            turn = turn + 1
            grounding.append(emos[num])
            question.append(context + role[turn%2] + ': ')
            answer.append(dialog[num])
            context = context + ' '+ role[turn%2] + ': ' + dialog[num]
    return grounding, question, answer, template_input, description

def get_commonsense_dialog(): # social-i-qa rewritten,chat 
    files = ['extra_data/Commonsense-Dialogues/data/train.json', 'extra_data/Commonsense-Dialogues/data/valid.json','extra_data/Commonsense-Dialogues/data/test.json']
    datasets = []
    for file in files:
        datasets.append(json.loads(open(file).read())) #9058 sessions, 51831 turns
    description = 'Task description: given the character background and context, predict the next utterance.'
    template_input = {}
    #prompts = DatasetTemplates('meta_woz')
    question = []
    answer = []
    grounding = []
    for dataset in datasets:
        for _,dialog in dataset.items():
            background = dialog['context']
            context = ''
            dial = dialog['turns']
            turn_num = 0
            for turn in dial:
                if (turn_num%2)==0:
                    speaker = dialog['speaker'].lower() #if 'speaker' in dialog else 'speaker1'
                else:
                    speaker = 'speaker2'
                if context == '':
                    context = speaker + ': '+turn
                else:
                    grounding.append(background)
                    question.append(context + ' ' + speaker + ': ')
                    answer.append(turn)
                    context = context + ' ' + speaker + ': '+turn
                turn_num = turn_num + 1
    return grounding, question, answer, template_input, description

####
#txt2sql

def get_spider():
    description ='Task description: translate the utterance into SQL format.'
    grounding = []
    answers = [] 
    template_input = {}
    #prompts = DatasetTemplates('spider')
    dataset = load_dataset("spider", split="train") #7000 success
    #dataset['db_id'] #domain 140
    query = dataset['query']  #'SELECT count(*) FROM head WHERE age  >  56'
    input = dataset['question'] #: input, question of natural language
    for q in query:
        answers.append(q.lower())
    return grounding, input, answers, template_input, description
    #dataset['query_toks']: 
    #dataset['query_toks_no_value']:
    #dataset['question_toks']:

def get_sparc(): # no prompt
    file = 'extra_data/sparc/train.json'  #3034, 12059 query-utt pairs
    description ='Task description: translate the utterance into SQL format.'
    grounding = []
    question = []
    answers = [] 
    template_input = {}
    dataset = json.loads(open(file).read())
    for case in dataset:
        query_step = case['interaction'] #list of utt, query
        for q in query_step:
            question.append(q['utterance'])
            answers.append(q['query'].lower())
        final_utt= case['final']['utterance']
        final_query= case['final']['query'].lower()
        question.append(final_utt)
        answers.append(final_query)

    return grounding, question, answers, template_input, description

####
#tod and user simulator# use space

##dataset = load_dataset("schema_guided_dstc8", split="train")

####
#rewrite

##dataset = load_dataset("task", split="train")

##dataset = load_dataset("canard", split="train")

##dataset = load_dataset("mudoco", split="train")

####
#summary

def get_samsum():
    path = 'data/pre_prompt/samsum.json'
    description ='Task description: summarize the dialog.'
    if os.path.exists(path):
        result = json.load(open(path, 'r'))
        grounding = result['grounding']
        question = result['query']
        answers= result['answer']
        template_input = result['template_input']
    else:
        grounding = []
        question = []
        answers = [] 
        template_input = {}
        dataset = load_dataset("samsum", split="train")  # success,14732
        prompts = DatasetTemplates('samsum')
        dial = dataset['dialogue']  #str name:utt,\r\n
        sum = dataset['summary'] # summary
        for num in range(len(sum)):
            for t in prompts.all_template_names:
                if t not in template_input:
                    template_input[t] = []
                prompt = prompts[t]
                template_input[t].append(prompt.apply(dataset[num]))
            question.append(dial[num].replace('\r\n',' '))
            answers.append(sum[num])
        result = {'grounding':grounding,'query':question,'answer':answers,'template_input':template_input}
        json.dump(result, open(path, 'w'),indent=2)
    return grounding, question, answers, template_input, description

def get_dialogsum(): # role Person1, Person2
    description ='Task description: summarize the dialog.'
    grounding = []
    question = []
    answers = [] 
    template_input = {}
    dataset = load_dataset("knkarthick/dialogsum", split="train")  # success,12460
    #prompts = DatasetTemplates('dialogsum')
    id = dataset['id'] 
    dial = dataset['dialogue']  #str name:utt,\n
    sum = dataset['summary'] # summary #Person2#,#Person1#
    domain = dataset['topic'] #domain 140
    for num in range(len(sum)):
        question.append(dial[num].replace('#','').replace('\n',' '))
        answers.append(sum[num].replace('#',''))
    return grounding, question, answers, template_input, description

def get_reading_comprehension(): # 10785, 180217 utts, not change qa into summary, can add unsup-training
    description ='Task description: Given the dialog, find out what person does placeholder refer to?'
    grounding = []
    question = []
    answers = [] 
    template_input = {}
    dir = 'extra_data/reading-comprehension/json'
    train_file = os.path.join(dir,'reading-comprehension-trn.json')
    data = json.loads(open(train_file).read())
    for case in data:
        context = ''
        dial = case['utterances']
        for turn in dial:
            context = context + turn['speakers'].replace('@','') + ': ' + turn['tokens'].replace('@','')
        grounding.append(context)
        question.append(case['query'].replace('@','') + ' Who is placeholder in this sentence?') #@placeholder can be substituted by answer
        answers.append(case['answer'].replace('@',''))            
               
    return grounding, question, answers, template_input, description # the question is crucial to get the answer

####
#fill

#dataset = load_dataset("restaurant8k", split="train")

#dataset = load_dataset("snips", split="train")  #in intent

#dataset = load_dataset("hwu64", split="train")  #in intent

def get_mams(): # 11186, to emo
    description ='Task description: Given the sentence, find out emotion towards certain aspect. '
    grounding = []
    question = []
    answers = [] 
    template_input = {}
    dir = 'extra_data/MAMS/data'
    term_dir = os.path.join(dir,'MAMS-ATSA')
    category_dir = os.path.join(dir,'MAMS-ACSA')
    train_path =  os.path.join(term_dir,'raw/train.xml')
    train_category_path =  os.path.join(category_dir,'raw/train.xml')
    from extra_data.MAMS.data_process.utils import parse_sentence_term,parse_sentence_category
    train_data = parse_sentence_term(train_path, lowercase=True)
    train_category_data = parse_sentence_category(train_category_path, lowercase=True)
    for case in train_data:
        grounding.append(case['text'])
        question.append('The attitude towards '+ case['term'] +' is')
        answers.append(case['polarity'])
    for case in train_category_data:
        grounding.append(case['text'])
        question.append('The attitude towards '+ case['category'] +' is')
        answers.append(case['polarity'])
    return grounding, question, answers, template_input, description

def get_aste(): # aspect sentiment triplet extraction, skip due to little data and process labor
    description ='Task description: Given the sentence, find out the feeling: postive, negative or neural.'
    grounding = []
    question = []
    answers = [] 
    template_input = {}
    dir = 'extra_data/Span-ASTE/data/triplet_data'
    files = ['14lap','14res','15res','16res'] #emo and act trained in daily dialog
    set = ['train.txt','test.txt','dev.txt'] #I charge it at night and skip taking the cord with me because of the good battery life .#### #### ####[([16, 17], [15], 'POS')], pos, neu, neg
    tmp='1'
    dataset = []
    for file in files:
        filename = os.path.join(dir,file,'train.txt')
        with open(filename, "r", encoding="utf-8") as fp:
            while tmp != "":
                tmp = fp.readline()
                if tmp != '':
                    dataset.append(tmp)
    #label_emo = {'POS':'positive','NEU':'neutral','NEG':'negitive'}
    for data in dataset:
        sentence = data.split('#### #### ####')[0]
        emos = data.split('#### #### ####')[1] # .split('),(')
        #for emo in emos:
        question.append(sentence.lower() + ' The feeling is ')
        #e = emo.split(']')
        if 'POS' in emos:
            answers.append('positive')
        elif 'NEG' in emos:
            answers.append('negitive')
        else:
            answers.append('neutral')
    return grounding, question, answers, template_input, description
    # relations can change into questions

def get_sentihood(): # can add verblizer
    description ='Task description: Given the sentence, find out emotion towards certain aspect. '
    grounding = []
    question = []
    answers = [] 
    template_input = {}
    dataset = load_dataset("bhavnicksm/sentihood", split="train")# success 2977
    id = dataset['id'] 
    senti = dataset['opinions']  #{'sentiment': 'Negative', 'aspect': 'price', 'target_entity': 'LOCATION1'}
    text = dataset['text'] # LOCATION1 ; 1 sent
    for num in range(len(senti)):
        for sent in senti[num]:
            grounding.append(text[num].lower())
            question.append('The '+ sent['aspect'] + ' of ' + sent['target_entity'].lower() +' is') # prompt needed
            answers.append(sent['sentiment'].lower())
    return grounding, question, answers, template_input, description

####
#commonsense

def get_commonsense_qa():
    path = 'data/pre_prompt/commonsense_qa.json'
    description ='Task description: choose the best word given the context and choice.'
    if os.path.exists(path):
        result = json.load(open(path, 'r'))
        grounding = result['grounding']
        question = result['query']
        answers= result['answer']
        template_input = result['template_input']
    else:
        grounding = []
        question = []
        answers = [] 
        template_input = {}
        dataset = load_dataset("commonsense_qa", split="train")  #success 9741 word understanding dataset
        prompts = DatasetTemplates('commonsense_qa')
        id = dataset['id'] 
        q=dataset['question']  
        qc = dataset['question_concept']  
        choice = dataset['choices'] #'text':list of 5 
        answer = dataset['answerKey'] #: 'A','B','C','D'
        all_answers = ['A','B','C','D','E']
        for num in range(len(q)):
            for t in prompts.all_template_names:
                if t not in template_input:
                    template_input[t] = []
                prompt = prompts[t]
                template_input[t].append(prompt.apply(dataset[num]))
            grounding.append(q[num])
            question.append('Which of the following words in ' + ', '.join(choice[num]['text']) + ' will you choose?')# +' describes the word '+ qc[num] + ' in the sentence?')
            answers.append(choice[num]['text'][all_answers.index(answer[num])])
        result = {'grounding':grounding,'query':question,'answer':answers,'template_input':template_input}
        json.dump(result, open(path, 'w'),indent=2)
    return grounding, question, answers, template_input, description

###dataset = load_dataset("alphanli", split="train")

def get_cosmos_qa():
    path = 'data/pre_prompt/cosmos_qa.json'
    description ='Task description: given the context and question, choose the answer from the choices.'
    if os.path.exists(path):
        result = json.load(open(path, 'r'))
        grounding = result['grounding']
        question = result['query']
        answers= result['answer']
        template_input = result['template_input']
    else:
        template_input = {}
        dataset = load_dataset("cosmos_qa", split="train")  #25262 success
        description = dataset.description
        id = dataset['id'] 
        grounding = dataset['context'] #some sentences
        prompts = DatasetTemplates('cosmos_qa')
        q = dataset['question'] 
        question = []
        label = dataset['label']#0-3 
        target = []
        target_wrong = []
        answer_data=[dataset['answer0'],dataset['answer1'],dataset['answer2'],dataset['answer3']]
        answers =[]
        for i in range(len(label)):
            for t in prompts.all_template_names:
                if t not in template_input:
                    template_input[t] = []
                prompt = prompts[t]
                template_input[t].append(prompt.apply(dataset[i]))
            answer = answer_data[label[i]][i]     
            answers.append(answer)
            question.append(q[i] + ' Choose from the following 4 choices: ' + dataset['answer0'][i] + ' ' +  dataset['answer1'][i] + ' ' + dataset['answer2'][i] + ' ' +  dataset['answer3'][i])
        result = {'grounding':grounding,'query':question,'answer':answers,'template_input':template_input}
        json.dump(result, open(path, 'w'),indent=2)
    return grounding, question, answers, template_input, description

##dataset = load_dataset("social_i_qa", split="train")  #social_interaction_qa #not connected

####
#emo
def get_reccon(task=0):
    description ='Task description: Given the context and utterance, find out the emotion or what causes the emotion. '
    grounding = []
    question = []
    answers = [] 
    template_input = {}
    file = 'extra_data/RECCON/data/original_annotation/dailydialog_train.json' #emo and act trained in daily dialog
    dataset = json.loads(open(file).read())  #8206
    for _,dial in dataset.items():
        context = ''
        for turn in dial[0]: 
            if task == 0:
                grounding.append(context)
                question.append(turn['speaker'] + ': ' + turn['utterance']) #+' What emotion does the sentence imply? ') 
                answers.append(turn['emotion'])
            if "expanded emotion cause span" in turn:
                emo_span = turn['expanded emotion cause span']
                if task == 1:
                    grounding.append(context)
                    question.append(turn['speaker'] + ': ' + turn['utterance'] +' What causes the emotion '+ turn['emotion'] +'? ') 
                    answers.append(', '.join(emo_span))
            context = context + ' ' + turn['speaker'] + ': ' + turn['utterance']    
    return grounding, question, answers, template_input, description
###dataset = load_dataset("emory", split="train")

def get_go_emotions(): # processed not ready, classification
    description ='Task description: Given the sentence, find out the emotion. '
    grounding = []
    question = []
    answers = [] 
    template_input = {}
    dir = 'extra_data/goemotions/data'
    file = os.path.join(dir,'train.tsv')
    data = pd.read_table(file, delimiter="\t",header=None)
    emo_file = 'extra_data/goemotions/data/emotions.txt'
    all_emo = pd.read_table(emo_file, delimiter="\t",header=None)
    all_emotions = all_emo.to_dict()[0]
    sentence = data.to_dict()[0]
    emo = data.to_dict()[1]
    for num in range(len(sentence)):
        grounding.append(sentence[num])
        question.append('') # use keyword emotion, feeling to regenerate the template
        # What emotion does the sentence imply
        if ',' not in emo[num]:
            answers.append(all_emotions[int(emo[num])])
        else:
            answers.append(', '.join(all_emotions[int(e)] for e in emo[num].split(',')))
    #dataset = load_dataset('go_emotions')
    return grounding, question, answers, template_input, description

def get_meld(task=0): # processed not ready
    description ='Task description: Given the sentence, find out the emotion or the type of emotion.'
    grounding = []
    question = []
    answers = [] 
    template_input = {}
    dir = 'extra_data/MELD/data/MELD/train_sent_emo.csv' #MELD_Dyadic
    data = pd.read_table(dir, delimiter=",")
    utt =  data.to_dict()['Utterance']
    #speaker =  data.to_dict()['Speaker']
    emo =  data.to_dict()['Emotion']
    senti =  data.to_dict()['Sentiment']
    for num in range(len(utt)):
        if task == 0:
            grounding.append(utt[num])
            question.append(' What kind of emotion does the sentence imply? ')
            answers.append(emo[num])
        if task == 1:
            grounding.append(utt[num])
            question.append(' What kind of emotion does the sentence imply, postive, negative or neutral? ')
            answers.append(senti[num])
    return grounding, question, answers, template_input, description

#knowledge-grounded
def get_kgdial(): # kg-qa
    description ='Task description: Given the external knowlegde, answer the question.'
    grounding = []
    question = []
    answers = [] 
    template_input = {}
    incar_file = 'extra_data/KG-Copy_Network/incar_conversations/train'
    #soccer_file = 'extra_data/KG-Copy_Network/soccar_conversations'
    kg_incar = 'extra_data/KG-Copy_Network/data/KG/incar'
    for file in os.listdir(incar_file):
        file_name = os.path.join(incar_file,file)
        conv = json.loads(open(file_name,'r').read())
        kg_name = os.path.join(kg_incar,file.replace('.json','_kg.txt'))
        final_kg = ''
        kg_list = open(kg_name,'r').read().split('\n')
        for ent in kg_list:
            if ent.split('\t')[0] in final_kg and (ent!=''):
                final_kg = final_kg + ' ' + ent.split('\t')[1]
            else:
                final_kg = final_kg + ' ' + ent
        final_kg = final_kg.replace('\t',' ').replace('_',' ')
        for key, qa in conv.items():
            if 'q' in key:
                grounding.append(final_kg)
                question.append(qa)
            else:
                answers.append(qa)

    return grounding, question, answers, template_input, description

#generation
def get_wiki_lingua(): # abstractive summary https://github.com/esdurmus/Wikilingua
    description ='Task description: Summarize the passage.'
    grounding = []
    template_input = {}
    dataset = load_dataset('GEM/wiki_lingua')#, split="train"
    #prompts = DatasetTemplates('GEM/wiki_lingua')
    train = dataset['train']
    question = train['source'] # sl = 'en'
    #tl = train['target_language']
    answers = train['target'] # tl = 'en'
    return grounding, question, answers, template_input, description
    
def get_dart(): #https://aclanthology.org/2021.naacl-main.37.pdf
    description ='Task description: Given the triplets, generate the sentence.'
    grounding = []
    template_input = {}
    question = []
    answers = []
    dataset = load_dataset('GEM/dart')#, split="train", 62659
    train = dataset['train']
    id = train['gem_id']
    triple = train['tripleset'] #.replace('\t',' '), perhaps it's hard to directly generate target from the triple, some extra information needed
    answer = train['target']
    for num in range(len(triple)):
        background = ''
        for tri in triple[num]:
            background = background +' '+ ' '.join(tri).lower() + ';'
        grounding.append(background)
        question.append('Generate a sentence using triplets.')
        answers.append(answer[num].replace('\t',' '))
    tree = train['subtree_was_extended'] # bool
    return grounding, question, answers, template_input, description
    
def get_totto(): # not use
    description ='Task description: Given the triplets, generate the sentence.'
    grounding = []
    template_input = {}
    question = []
    answers = []
    dataset = load_dataset('GEM/totto')#, split="train"
    train = dataset['train']
    description = train.description
    title = train['table_page_title']
    input = train['linearized_input'] #sp tokens <page_title> <section_title>,etc, need parsing
    annotation = train['sentence_annotations']
    for num in range(len(input)):
        tmp = pd.read_xml(input[num])
    answers = train['target']
    return grounding, question, answers, template_input, description

def get_e2e_nlg():
    description ='Task description: Given the key value pairs, generate the sentence.'
    grounding = []
    template_input = {}
    question = []
    answers = []
    dir = 'extra_data/e2e-cleaning-master/cleaned-data/train-fixed.no-ol.csv'
    tmp = pd.read_table(dir, delimiter=",")
    data = tmp.to_dict()
    mr = data['mr']
    ref = data['ref']
    for num in range(len(mr)):
        grounding.append(mr[num].replace('[',': ').replace(']','')) # \\u[0]{2}([0-9a-z]{2})
        question.append('Generate a sentence using key value pairs.')
        answers.append(ref[num])
    return grounding, question, answers, template_input, description

def get_common_gen():
    path = 'data/pre_prompt/common_gen.json'
    description ='Task description: Given the triplets, generate the sentence.'
    if os.path.exists(path):
        result = json.load(open(path, 'r'))
        grounding = result['grounding']
        question = result['query']
        answers= result['answer']
        template_input = result['template_input']
    else:
        grounding = []
        template_input = {}
        question = []
        answers = []
        dataset = load_dataset('GEM/common_gen')#, split="train",triple to text
        prompts = DatasetTemplates('common_gen')
        train = dataset['train']
        concept_set_id = train['concept_set_id'] # no use
        concepts = train['concepts']
        for num in range(len(concepts)):
            for t in prompts.all_template_names:
                if t not in template_input:
                    template_input[t] = []
                prompt = prompts[t]
                template_input[t].append(prompt.apply(train[num]))
            grounding.append(' '.join(concepts[num]))
            question.append('What sentence can you generate with those words?')
        answers = train['target']
        result = {'grounding':grounding,'query':question,'answer':answers,'template_input':template_input}
        json.dump(result, open(path, 'w'),indent=2)
    return grounding, question, answers, template_input, description

def get_xlsum():
    description ='Task description: Summarize the passage.'
    grounding = []
    template_input = {}
    question = []
    answers = []
    dataset = load_dataset('GEM/xlsum','english')# bbc, split="train"
    train = dataset['train']
    title = train['title'] 
    source_text = train['text']
    target = train['target']
    for num in range(len(source_text)):
        source = source_text[num].replace('\n\n',' ')
        if len(source)<2000:
            grounding.append(source)
            question.append('Summarize the passage.')
            answers.append(target[num])
    return grounding, question, answers, template_input, description

def get_web_nlg(): # world knowlegde
    description ='Task description: Given the domain and triplets, generate the sentence.'
    grounding = []
    template_input = {}
    question = []
    answers = []
    dataset = load_dataset('GEM/web_nlg','en')#, split="train"
    train = dataset['train']
    category = train['category'] # domain:16 
    input = train['input'] #replace('| ',' ')
    for num in range(len(category)):
        grounding.append('domain: '+category[num] + ' triple: ' + input[num][0].replace('| ',' '))
        question.append('What sentence can you generate with the domain and the triple?')
    answers = train['target']
    return grounding, question, answers, template_input, description

def get_xwikis(): # too large, contains 987228 cases, can be sampled to contain cases that have less than 2000 characters(189784)
    """
    dataset = load_dataset('GEM/xwikis','en-de')#, split="train", no en-fr available
    train = dataset['train']
    summary = train['src_summary']
    title = train['src_title']
    document = train['src_document']
    """
    doc_dir = 'extra_data/xwikis/en_documents.txt'
    sum_dir = 'extra_data/xwikis/en_summaries.txt'
    title_dir = 'extra_data/xwikis/en_titles.txt'
    docs = open(doc_dir,'r').read().split('\n')
    sums = open(sum_dir,'r').read().split('\n')
    titles = open(title_dir,'r').read().split('\n')
    description ='Task description: Given the title and passage, generate the summary.'
    grounding = []
    template_input = {}
    question = []
    answers = []
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    for num in range(len(docs)): # len(docs)
        if len(docs[num])<2000: # 1800
            grounding.append('title: '+titles[num] + ' passage: ' + docs[num])
            question.append('Summarize the passage.')
            answers.append(sums[num])
    return grounding, question, answers, template_input, description

def get_ketod(task=0): # 
    description ='Task description: Given the context, generate the next utterance or find the user intent or user state.'
    grounding = []
    template_input = {}
    question = []
    answers = []
    dir = 'extra_data/ketod-main/ketod_release/train.json'
    data = json.loads(open(dir,'r').read())
    for dial in data:
        context = ''
        background = dial['entity_passages']
        for turn in dial['turns']:
            if context!='':
                if turn['speaker'] == 'USER':
                    structure_info = turn['frames'][0]['state']
                    if task == 0:
                        #state_prompts
                        if structure_info['slot_values']=={}:
                            #answers.append('None.')
                            states = 'None.'
                        else:
                            states = ''
                            for k,v in structure_info['slot_values'].items():
                                states = states + ' ' + k + ' ' + v[0] + ','
                        input = random.sample(state_prompts,k=1)[0] 
                        if isinstance(input,list):
                            grounding.append(input[0] + context + ' ' + turn['speaker'].lower() + ': ' +turn['utterance'])
                            question.append(input[1])
                            answers.append(states.replace('_',' '))
                        else:
                            grounding.append(context + ' ' + turn['speaker'].lower() + ': ' +turn['utterance'])
                            question.append(input)
                            answers.append(states.replace('_',' '))
                    if task == 2:
                        grounding.append('context: '+context)
                        question.append('What is the user intent?')
                        answers.append(structure_info['active_intent'])
                turn_grounding = ''
                for k,v in background.items(): # entity name based knowlegde enhance
                    if k.split(':')[-1] in turn['utterance']:
                        for par in v:
                            if par[0] in turn['utterance']:
                                turn_grounding = turn_grounding + ' ' + ' '.join(par[1:])
                if task == 1:
                    grounding.append(turn_grounding)
                    question.append('context: '+context + ' ' + turn['speaker'].lower() + ': ')
                    answers.append(turn['utterance'])
            context = context + ' ' + turn['speaker'].lower() + ': ' + turn['utterance']
    return grounding, question, answers, template_input, description

def get_fusedchat(): # dst is the same as multiwoz, so only trained unsupervisedly
    task = 'task-oriented dialog' # chat
    description ='Task description: Given the domain and triplets, generate the sentence.'
    grounding = []
    template_input = {}
    question = []
    answers = []
    dir = 'extra_data/FusedChat-main/data/appended_lexicalized.json'
    prepend_dir = 'extra_data/FusedChat-main/data/prepended_lexicalized.json'
    data = json.loads(open(dir,'r').read())
    prepend_data = json.loads(open(prepend_dir,'r').read())
    all_data = []
    for _,v in data['train'].items():
        all_data.append(v)
    for _,v in prepend_data['train'].items():
        all_data.append(v)
    role = ['user','system']
    for dial in all_data:
        context = ''
        for turn_num in range(len(dial['log'])):
            turn = dial['log'][turn_num]
            if context!='':
                question.append(context + ' ' + role[turn_num%2] + ': ')
                answers.append(turn['text'])
            context = context + ' ' + role[turn_num%2] + ': ' + turn['text']
    return grounding, question, answers, template_input, description

def get_dst_string(dst):
    dst_str = ''
    flag = 0
    for domain,kv in dst.items():
        if domain != "DEFAULT_DOMAIN":
            dst_str = dst_str + ' ' + 'domain: ' + domain # if flag==0
        if kv != {}:
            for key,value in kv.items():
                if value!=[]:
                    flag = 1
                    dst_str = dst_str + ' ' + key.lower() + ' ' + ' '.join(value) + ', '
    if dst_str.endswith(', '):
        dst_str = dst_str[:-2]
    return dst_str, flag

def get_multiwoz_dst(no_glue=True):
    if no_glue:
        filename = 'data/pre_prompt/multiwoz_dst10.json'
        AUG_NUM = 2
    else:
        filename = 'data/pre_prompt/multiwoz_dst.json'
        AUG_NUM = 1
    tmp = '1'
    data = []
    choice_prompt = json.load(open('data/new_prompts/choice_final.json', 'r'))
    qa_prompt = json.load(open('data/new_prompts/answer_final.json', 'r'))
    with open(filename, "r", encoding="utf-8") as fp:
        while tmp != "":
            tmp = fp.readline()
            if tmp != '' and tmp!='1':
                turn = json.loads(tmp)
                turn['dialogue'] = turn['dialogue'].replace("[domain]", "Domain ").replace("[slot]", "Slot ")
                if "[PVs]" in turn['dialogue']:
                    prompts = random.sample(choice_prompt, AUG_NUM)
                    text = turn['dialogue'].replace("[PVs]", ", for example: ")
                    for prompt in prompts:
                        if isinstance(prompt, list):
                            turn['dialogue'] = text + '. ' +  prompt[1]
                        else:
                            turn['dialogue'] = text + '. ' +  prompt
                        data.append(copy.deepcopy(turn))
                else:
                    prompts = random.sample(qa_prompt, AUG_NUM)
                    text = turn['dialogue']
                    for prompt in prompts:
                        if isinstance(prompt, list):
                            turn['dialogue'] = text + '. ' +  prompt[1]
                        else:
                            turn['dialogue'] = text + '. ' +  prompt
                        data.append(copy.deepcopy(turn))
    return data

def get_space_single(no_glue=True, pseudo_labels=True):
    dir = 'data/pre_train/AnPreDial/single_turn'
    # intent_dataset = ['banking','clinc','hwu']
    # slot_dataset = ['atis','CrossNER',] # 'requestedSlots'
    intent_sample = [] # 115873, 226576(augmented)
    state_sample = []  # 109348
    dialoglue = ['banking', 'dstc8_sgd', 'hwu', 'restaurant8k', 'top', 'clinc']
    intent_non_aug = ['inform', 'inform', 'request', 'greeting', 'other', 'greeting', 'bye', 'find']
    aug_num = 2
    intent_prompts_dg = json.load(open('data/new_prompts/intentsimplified_selected_dialoglue.json','r'))
    state_prompts_dg = json.load(open('data/new_prompts/statesimplified_selected_dialoglue.json','r'))
    for file in os.listdir(dir): # single_turn
        if file not in dialoglue:
            if file in dialoglue:
                aug_num = 10
            else:
                aug_num = 2
            intent_grounding = []
            intent_question = []
            intent_answers = []
            state_grounding = []
            state_question = []
            state_answers = []
            name = os.path.join(dir,file)
            data = json.loads(open(os.path.join(name,'train.json'),'r').read())
            for _,case in data.items():
                turn = case['turns']
                if (len(turn)==2) and (turn[0]['role'] == 'sys' or turn[0]['role'] == 'system'): #file not in  ['restaurant8k','multiwoz_coco']:
                    turn = case['turns'][1]
                else:
                    turn = case['turns'][0] 
                dst = ''
                all_intents = []
                for domain, isv in turn['label'].items():
                    #if turn['role'] == 'user':
                    #    if domain not in dst:
                    #        dst [domain] = {}
                    if (domain != "DEFAULT_DOMAIN") and (domain != "conll2003"):
                        dst = dst + ' ' + domain
                    for intent,sv in turn['label'][domain].items():
                        if intent != 'DEFAULT_INTENT':
                            #if file in intent_dataset:
                            all_intents.append(intent)                 
                        # else: # slot data
                        if sv!={}:
                            for slot,value in sv.items():
                                dst = dst + ' '+slot.lower() + ' ' + value[0]['value'] +  ','
                if all_intents!=[] and file!='atis':
                    if ((''.join(all_intents).lower()) in intent_non_aug) or file=='top':
                        inputs = random.sample(intent_prompts,k=1) 
                    else:
                        inputs = random.sample(intent_prompts,k=aug_num) 
                    for input in inputs:
                        if isinstance(input,list):
                            intent_question.append(input[1].replace('user',turn['role']))
                            intent_grounding.append(turn['role'] + ': ' +turn['text']) # input[0] + ' '+
                            intent_answers.append((','.join(all_intents)).replace('_',' ').replace('-',' ').replace('atis ',' ').lower()) 
                        else:
                            intent_question.append(input.replace('user',turn['role']))
                            intent_grounding.append(turn['role'] + ': ' +turn['text'])
                            intent_answers.append((','.join(all_intents)).replace('_',' ').replace('-',' ').replace('atis ',' ').lower())  
                #if dst != {} and dst != {'DEFAULT_DOMAIN':{}}:
                #    if turn['role'] == 'user':
                #        state_grounding.append(turn['role'] + ': ' +add_endmark(turn['text']))
                #        state_question.append('What is the '+ turn['role'] + ' state?')
                #        state_answers.append(get_dst_string(dst))
                if dst != '':
                    #state_grounding.append(turn['role'] + ': ' +turn['text'])
                    #state_question.append('What is the '+ turn['role'] + ' state?')
                    #state_answers.append(dst[:-1] if dst[-1]==',' else dst)
                    inputs = random.sample(state_prompts,k=aug_num) 
                    for input in inputs:
                        if isinstance(input,list):
                            state_grounding.append(turn['role'] + ': ' +turn['text']) # input[0] + ' ' +
                            state_question.append(input[1].replace('user',turn['role']))
                            state_answers.append(dst[:-1] if dst[-1]==',' else dst)
                        else:
                            state_grounding.append(turn['role'] + ': ' +turn['text'])
                            state_question.append(input.replace('user',turn['role']))
                            state_answers.append(dst[:-1] if dst[-1]==',' else dst)
            if intent_grounding!= []:
                intent_sample.append((intent_grounding,intent_question,intent_answers))
            if state_grounding!= []:
                state_sample.append((state_grounding,state_question,state_answers))

        # if trained without 10 data, do not add this part
        if not no_glue:
            if file in dialoglue:
                if file in ['banking', 'hwu', 'clinc', 'top']:
                    filename = file + '_intent' if file=='top' else file
                    data = json.load(open('data/pre_prompt/'+filename+'_train.json','r'))
                    intent_grounding = []
                    intent_question = []
                    intent_answers = []
                    for case in data:
                        intent_question.append(case[0])
                        intent_grounding.append('')
                        intent_answers.append(case[1].replace('_',' ').replace('-',' ').lower())
                    intent_sample.append((intent_grounding,intent_question,intent_answers))
                if file in ['dstc8_sgd', 'restaurant8k', 'top']:
                    filename = file + '_slot' if file=='top' else file
                    data = json.load(open('data/pre_prompt/'+filename+'_10.json','r'))
                    state_grounding = []
                    state_question = []
                    state_answers = []
                    for case in data:
                        if case[1]!='':
                            state_question.append(case[0])
                            state_grounding.append('')
                            state_answers.append(case[1].replace('-',' ').lower())
        else:
            if file in ['banking', 'hwu', 'clinc', 'top']: # if do not add this
                filename = file + '_intent' if file=='top' else file
                data = json.load(open('data/pre_prompt/'+filename+'_10.json','r'))
                intent_grounding = []
                intent_question = []
                intent_answers = []
                for case in data:
                    intent_question.append(case[0])
                    intent_grounding.append('')
                    intent_answers.append(case[1].replace('_',' ').replace('-',' ').lower())
                if pseudo_labels:
                    AUG_NUM = 3 # different from the outer aug_num
                    pseudo_data = json.load(open('data/pseudo_labels/pseudo_'+filename+'.json','r'))
                    for case in pseudo_data:
                        prompt = random.sample(intent_prompts_dg, AUG_NUM)
                        for p in prompt:
                            intent_question.append(p[0] + case[0] + ' ' + p[1])
                            intent_grounding.append('')
                            intent_answers.append(case[1].replace('_',' ').replace('-',' ').lower())
                intent_sample.append((intent_grounding,intent_question,intent_answers))
            if file in ['dstc8_sgd', 'restaurant8k', 'top']:
                filename = file + '_slot' if file=='top' else file
                data = json.load(open('data/pre_prompt/'+filename+'_10.json','r'))
                state_grounding = []
                state_question = []
                state_answers = []
                for case in data:
                    if case[1]!='':
                        state_question.append(case[0])
                        state_grounding.append('')
                        state_answers.append(case[1].replace('-',' ').lower()) # replace('_',' '), keep the same format as dataset
                if pseudo_labels and file!='dstc8_sgd': # need to be removed next time
                    AUG_NUM = 3 # different from the outer aug_num
                    pseudo_data = json.load(open('data/pseudo_labels/pseudo_'+file+'.json','r'))
                    for case in pseudo_data:
                        prompt = random.sample(state_prompts_dg, AUG_NUM)
                        for p in prompt:
                            state_question.append(p[0] + case[0] + ' ' +  p[1])
                            state_grounding.append('')
                            state_answers.append(case[1].replace('_',' ').replace('-',' ').lower())
                if state_grounding!= []:
                    state_sample.append((state_grounding,state_question,state_answers))
    # add multi-woz dst in this file, for simplicity not added in multi-turn
    # the task should actually be choice or qa, as the prompts and input-output format
    data = get_multiwoz_dst(no_glue)
    state_grounding = []
    state_question = []
    state_answers = []
    for case in data:
        # if case[1]!='':
        state_question.append(case['dialogue'])
        state_grounding.append('')
        state_answers.append(case['state'])
    if not no_glue:
        json.dump(intent_sample, open('data/pre_train/prompts/space_single_intent.json', 'w'), indent=2) # use with encoded_data_all, 6
        json.dump(state_sample, open('data/pre_train/prompts/space_single_state.json', 'w'), indent=2) # 8
        #json.dump(intent_sample, open('data/pre_train/space_single_intent.json', 'w'),indent=2)
        #json.dump(state_sample, open('data/pre_train/space_single_state.json', 'w'),indent=2)
    else:
        json.dump(intent_sample, open('data/pre_train/prompts/space_single_intent_glue.json', 'w'),indent=2)# 6
        json.dump(state_sample, open('data/pre_train/prompts/space_single_state_glue.json', 'w'),indent=2) # 8
    return intent_sample, state_sample    

def get_space_multi():
    dialoglue = ['MultiWOZ2.2', 'SGD']
    save_file = 'data/pre_prompt/space_multi.json'
    aug_num = 2
    intent_non_aug = ['SwDA', 'inform', 'request', 'greeting', 'other', 'greeting', 'bye', 'find']
    if os.path.exists(save_file):
        result = json.loads(open(save_file,'r').read())
        intent_sample = result['intent_sample']
        state_sample = result['state_sample']
        response_sample = result['response_sample']
    else:
        dir_multi = 'data/pre_train/AnPreDial/multi_turn'
        intent_sample = [] # origin:2*1179370 2358740,1858469 can add qa prompt as augmentation
        # do not augment for common intent
        
        state_sample = [] # 2106966, 1547586  can add qa prompt as augmentation
        response_sample = [] # 2369618
        for file in os.listdir(dir_multi):
            intent_grounding = []
            intent_question = []
            intent_answers = []
            state_grounding = []
            state_question = []
            state_answers = []
            response_grounding = []
            response_question = []
            response_answers = []
            name = os.path.join(dir_multi,file)
            data = json.loads(open(os.path.join(name,'train.json'),'r').read())
            for _,dial in data.items():
                contexts = []
                dst = {}
                last_state = ''
                for turn in dial['turns']:
                    # get response_sample
                    context = get_context(contexts)
                    if context != '':
                        if turn['role'] == 'user': # user simulation
                            response_grounding.append('')
                            if len(context)>2000:
                                response_question.append(context[-2000:] + ' ' + turn['role'] + ': ')
                            else:
                                response_question.append(context + ' ' + turn['role'] + ': ')
                            response_answers.append(add_endmark(turn['text']))
                        elif turn['role'] == 'system':
                            if dst != {} and dst != {'DEFAULT_DOMAIN':{}}:
                                response_grounding.append('User info: ' + get_dst_string(dst)[0]) # user state
                            else:
                                response_grounding.append('')
                            if len(context)>2000:
                                response_question.append(context[-2000:] + ' ' + turn['role'] + ': ')
                            else:
                                response_question.append(context + ' ' + turn['role'] + ': ')
                            response_answers.append(add_endmark(turn['text']))
                    all_intents = []
                    for domain, isv in turn['label'].items():
                        #if domain != "DEFAULT_DOMAIN":
                        if turn['role'] == 'user':
                            if domain not in dst:
                                dst [domain] = {}
                        for intent,sv in turn['label'][domain].items():
                            if intent != 'DEFAULT_INTENT':
                                if intent not in (''.join(all_intents)):
                                    all_intents.append(intent)
                            else: # slot data
                                if sv!={} and turn['role'] == 'user':
                                    for slot,value in sv.items():
                                        if slot not in dst[domain]:
                                            dst[domain][slot] = []
                                        if (value[0]['value']!='') and (value[0]['value'] not in dst[domain][slot]):
                                            dst[domain][slot].append(value[0]['value'])                
                    if all_intents!=[]:
                        if (''.join(all_intents).lower()) in intent_non_aug:
                            inputs = random.sample(intent_prompts,k=1)
                        else: 
                            inputs = random.sample(intent_prompts,k=aug_num) 
                        for input in inputs:
                            if isinstance(input,list):
                                intent_question.append(input[1].replace('user',turn['role']))
                                intent_grounding.append(turn['role'] + ': ' +add_endmark(turn['text'])) # input[0] + ' '+
                                intent_answers.append((', '.join(all_intents)).replace('_',' ').replace('-',' ').lower()) 
                            else:
                                intent_question.append(input.replace('user',turn['role']))
                                intent_grounding.append(turn['role'] + ': ' +add_endmark(turn['text']))
                                intent_answers.append((', '.join(all_intents)).replace('_',' ').replace('-',' ').lower()) 
                    if dst != {} and dst != {'DEFAULT_DOMAIN':{}}:
                        if turn['role'] == 'user':
                            d_str, has_sv = get_dst_string(dst)
                            if has_sv:
                                #state_grounding.append(context + turn['role'] + ': ' +add_endmark(turn['text']))
                                #state_question.append('What is the '+ turn['role'] + ' state?')
                                #state_answers.append(d_str)
                                inputs = random.sample(state_prompts,k=aug_num) 
                            else:
                                inputs = random.sample(state_prompts,k=(aug_num-1))
                            for input in inputs:
                                if isinstance(input,list):
                                    state_grounding.append(((' Previous user info: ' + last_state + ' . ') if last_state!='' else '') + input[0] + ' ' + context + ' ' + turn['role'] + ': ' +turn['text'] )
                                    state_question.append(input[1])
                                    state_answers.append(d_str)
                                else:
                                    state_grounding.append(((' Previous user info: ' + last_state + ' . ') if last_state!='' else '') + context + ' ' + turn['role'] + ': ' +turn['text'] )
                                    state_question.append(input)
                                    state_answers.append(d_str)
                            last_state = d_str
                    contexts.append(turn)
            if intent_grounding!= []:
                intent_sample.append((intent_grounding,intent_question,intent_answers))
            if state_grounding!= [] and (file not in dialoglue):
                state_sample.append((state_grounding,state_question,state_answers))
            if response_grounding!= []:
                response_sample.append((response_grounding,response_question,response_answers))
        result = {'intent_sample':intent_sample,'state_sample':state_sample,'response_sample':response_sample}
        json.dump(result, open(save_file, 'w'),indent=2)
    json.dump(intent_sample, open('data/pre_train/prompts/space_multi_intent.json', 'w'),indent=2) # 15 datasets
    json.dump(state_sample, open('data/pre_train/prompts/space_multi_state.json', 'w'),indent=2) # 10 datasets
    json.dump(response_sample, open('data/pre_train/prompts/space_multi_response.json', 'w'),indent=2) # 18 datasets
    return intent_sample, state_sample,response_sample

def get_space_un(): # 21
    dir = 'data/pre_train/UnPreDial'
    question = []
    answers = []
    for file in os.listdir(dir):
        if file!='.DS_Store':
            data = json.loads(open(os.path.join(dir, file, 'train.json'),'r').read())
            for _,dial in data.items():
                contexts = []
                for turn_num in range(len(dial['turns'])):
                    turn = dial['turns'][turn_num]
                    context = get_context(contexts)
                    if context != '':
                        #if len(context)>1500:
                        #    question.append(context[-1500:])
                        #else:
                        question.append(context + ' ' + turn['role'] + ': ')
                        answers.append(turn['text'])
                    contexts.append(turn) # = context + ' ' + turn['role'] + ': ' +turn['text']
    json.dump((question,answers), open('data/pre_train/prompts/space_un.json', 'w'),indent=2)    

def check():
    #samples = json.load(open('data/pre_train/data_overview.json','r'))
    test = json.load(open('data/pre_train/unsup_encoded.json','r'))
    """
    dataset = load_dataset('banking77')
    template_input= {}
    prompts = DatasetTemplates('banking77')
    case = dataset['train'][0]
    for t in prompts.all_template_names:
        if t not in template_input:
            template_input[t] = []
        prompt = prompts[t]
        result = prompt.apply(case)
        template_input[t].append(result)
    """
    print(1)

def get_dataset_info(): # total num:45
    # remove the persona file in the data/pre_train_prompt file to probe few-shot and zero-shot ability of persona
    multitask = ['woi','dailydialog','reccon','meld','ketod']
    data_info = {
    'doc_qa':['squad','quac','narrativeqa','race'], # change 'doqa' to response_generation due to its context useage
    'dialqa':['dream','dialog_re', 'ddrel', 'friends_qa','molweni', 'reccon1', 'reading_comprehension', 'woi0'],
    # woi0 is a searching task, prompt? 
    # ddrel, dialog_re: the task is to discover relationship between speakers
    # some of the above case can be changed to multiple choices
    'text2sql':['spider', 'sparc'], 
    # grounded on various information
    'response_generation':['doqa', 'personachat', 'metawoz', 'empathetic_dialogues', 'ketod1', 'fusedchat','mutual', 'woi1', 'commonsense_dialog'], 
    #A，B： personachat, metawoz, empathetic_dialogues
    'summary':['samsum', 'dialogsum', 'xlsum', 'xwikis', 'wiki_lingua'],
    #dialog summary: samsum, dialogsum
    'intent': ['dailydialog0'],
    'dst':['ketod0'],
    # commonsense_qa
    'multiple_choice':['commonsense_qa', 'cosmos_qa', 'meld1'], # 'meld1': multiple choice in emotional
    'emotional':['dailydialog1', 'go_emotions', 'meld0', 'sentihood', 'mams', 'aste',  'reccon0'], 
    # emo_list for reccon and dailydialog= ['no emotion', 'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
    # meld1,aste: positive, negative neural, sentihood2
    'kg':['kgdial'], 
    'nlg':['web_nlg', 'dart', 'e2e_nlg', 'common_gen']
    # triple web_nlg
    }
    json.dump(data_info, open('data/pre_train/prompts/all_tasks.json', 'w'),indent=2)   

if __name__ == "__main__":

    ##doc_qa
    #get_squad()
    #get_quac()
    #get_narrativeqa()
    #get_race()
    #get_doqa()

    ##dialqa
    #get_dream()
    #get_dialog_re()
    #get_ddrel()
    #get_friends_qa()
    #get_molweni()
    #get_mutual()

    ##text2sql
    #get_sparc()
    #get_spider()

    ##chat
    #get_dailydialog(FSB=True)
    #get_personachat(True)
    #get_metawoz()
    #get_empathetic_dialogues()

    ##summary
    #get_samsum()
    #get_dialogsum()
    #get_reading_comprehension()

    ##slot_filling in dialogzoo, senti classification indeed
    #get_sentihood()
    #get_aste()
    #get_mams()

    ##commonsense qa
    #get_commonsense_qa()
    #get_cosmos_qa()

    ##emotional conversation
    #get_go_emotions()
    #get_meld()
    #get_reccon()

    ##kgdial
    #get_kgdial()
    #get_woi(task='search')
    #get_commonsense_dialog()

    ##generation
    #get_wiki_lingua()
    #get_dart()
    #get_totto()#
    #get_e2e_nlg()
    #get_common_gen()
    #get_xlsum()
    #get_web_nlg()
    #get_xwikis()

    ##fuse task
    #get_ketod()
    #get_fusedchat()

    ## space
    # if no_glue=False, set pseudo_labels=False
    #get_space_single(no_glue=False, pseudo_labels=False)
    #get_space_multi()
    get_space_un()

    ## other useful scripts in getting data
    #check()
    #get_dataset_info()