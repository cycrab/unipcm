"""
Dataset class
"""
import json
import logging
import os
import random
from tqdm import tqdm
from multiprocessing import Pool

from torch.utils.data import Dataset


class LazyDataset(Dataset):
    """
    Lazy load dataset from disk.
    Each line of data file is a preprocessed example.
    """

    def __init__(self, tokenizer, hparams, transform=lambda s: json.loads(s)): # data_file,
        """
        Initialize lazy dataset.

        By default, loading .jsonl format.

        :param data_file
        :type str

        :param transform
        :type callable
        """
        self.test = 0
        self.train_file = 'data/pre_train/train_encoded_all_1.json'
        self.eval_file = 'data/pre_train/eval_encoded_all_1.json'
        self.tokenizer = tokenizer
        self.max_seq_length = hparams.max_seq_length
        self.dataset = []

        file = {'train':self.train_file, 'eval':self.eval_file,}
        if not os.path.exists(self.train_file):
            self.load_dataset(tokenizer, hparams)
        else:
        #    self.transform = transform
            #self.reader = reader
            self.offsets = [0]
            with open(file[hparams.type], "r", encoding="utf-8") as fp:
                while fp.readline() != "":
                    self.offsets.append(fp.tell())
            self.offsets.pop()
            self.fp = open(file[hparams.type], "r", encoding="utf-8")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        self.fp.seek(self.offsets[idx], 0)
        sample = self.transform(self.fp.readline().strip())
        #if self.reader.with_mlm:
        # sample = self.reader.create_masked_lm_predictions(sample)
        return sample

    def process(self,i): #start, stop):
        #for i in range(start, stop):
        self.test = self.test + i
        self.dataset.append({'input':tokenizer.encode(self.grounding[i] + ' ' + self.query[i])[-self.max_seq_length:]
        ,'output':tokenizer.encode(self.answers[i])[-self.max_seq_length:]})
        return

    def load_dataset(self, tokenizer, hparams):
        logging.info('loading dataset')
        #if os.path.exists(path):
        #dataset = json.load(open(path,'r'))
        dir = 'data/pre_train/prompts'
        all_prompts = json.loads(open(os.path.join(dir,'all_prompts.json'), 'r').read())
        all_tasks = json.loads(open(os.path.join(dir,'all_tasks.json'), 'r').read())
        all_tasks_list = [k for k in all_tasks]
        emotion_prompts = json.loads(open('data/new_prompts/emotion_final.json', 'r').read())
        summary_prompts = json.loads(open('data/new_prompts/summary_final.json', 'r').read())
        answer_prompts = json.loads(open('data/new_prompts/answer_final.json', 'r').read())
        choice_prompts = json.loads(open('data/new_prompts/choice_final.json', 'r').read())
        generation_prompts = json.loads(open('data/new_prompts/generated_final.json', 'r').read())
        txt2sql_prompts = json.loads(open('data/new_prompts/key information_final.json', 'r').read())
        self.debug = []
        for file in os.listdir(dir):
            print(file)
            data = json.loads(open(os.path.join(dir,file), 'r').read())
            if 'intent' in file or 'state' in file or 'response' in file : # slow, to use multi-thread

                for subdata in data:
                    self.grounding = subdata[0]
                    self.query = subdata[1]
                    self.answers = subdata[2]
                    def process(self,i): #start, stop):
                        #for i in range(start, stop):
                        self.dataset.append({'input':tokenizer.encode(grounding[i] + ' ' + query[i])[-hparams.max_seq_length:]
                        ,'output':tokenizer.encode(answers[i])[-hparams.max_seq_length:]})
                        return
                    if 'response' in file:
                        for i in tqdm(range(len(self.query))):
                            if i < 10:
                               self.debug.append({'input':(self.grounding[i] + ' ' + self.query[i]),'output':self.answers[i], 'task': all_tasks_list.index('response_generation')})#'task':'response'         
                            #self.dataset.append({'input':tokenizer.encode(self.grounding[i] + ' ' + self.query[i])[-hparams.max_seq_length:], 'output':tokenizer.encode(self.answers[i])[-hparams.max_seq_length:], 'task': all_tasks_list.index('response_generation')})
                            self.dataset.append({'input':self.grounding[i] + ' ' + self.query[i][-hparams.max_seq_length:]
                            ,'output':self.answers[i][-hparams.max_seq_length:], 'task': all_tasks_list.index('response_generation')})
                    elif 'state' in file:
                        for i in tqdm(range(len(self.query))):
                        #   i = range(len(self.query))
                        #for n in range(0,len(query), int(len(query)/10)):
                        #    stop = n + int(len(query)/10) if n + int(len(query)/10) <= len(query) else len(query)
                        #    threading.Thread(target = process, args = (n, stop)).start()  
                        #pool = Pool(10)            
                        #pool.map(self.process,i) 
                        #pool.close()
                        #pool.join()
                            if i < 10:
                               self.debug.append({'input':(self.grounding[i] + ' ' + self.query[i]),'output':self.answers[i],'task': all_tasks_list.index('dst')}) 
                            #self.dataset.append({'input':tokenizer.encode(self.grounding[i] + ' ' + self.query[i])[-hparams.max_seq_length:] , 'output':tokenizer.encode(self.answers[i])[-hparams.max_seq_length:], 'task':all_tasks_list.index('dst')})
                            self.dataset.append({'input':self.grounding[i] + ' ' + self.query[i][-hparams.max_seq_length:] , 'output':self.answers[i][-hparams.max_seq_length:], 'task':all_tasks_list.index('dst')})
                    elif 'intent' in file:
                        for i in tqdm(range(len(self.query))):
                            if i < 10:
                               self.debug.append({'input':(self.grounding[i] + ' ' + self.query[i]),'output':self.answers[i]}) 
                            self.dataset.append({'input':tokenizer.encode(self.grounding[i] + ' ' + self.query[i])[-hparams.max_seq_length:] #'task':'intent', #'task':'state'
                            ,'output':tokenizer.encode(self.answers[i])[-hparams.max_seq_length:], 'task':all_tasks_list.index('intent')})
            elif 'space_un' in file :
                pass
                #def process1(start, stop):
                #    for i in range(start, stop):
                #        if i < 10:
                #            self.debug.append({'input':data[0][i],'output':data[1][i]})
                #        self.dataset.append({'input':tokenizer.encode(data[0][i])[-hparams.max_seq_length:],
                #        'output':tokenizer.encode(data[1][i])[-hparams.max_seq_length:]}) # 'task':'response'
                #for i in tqdm(range(int(len(data[0])/10))): # use 1/10 unannoted dialogs
                #for n in range(0,len(data), 10):
                #    intent_encoded = tokenizer.encode(data[0][i])
                #    stop = n + int(len(query)/10) if n + int(len(query)/10) <= len(query) else len(query)
                #    threading.Thread(target = process1, args = (n, stop)).start() 
                #    self.dataset.append({'input':tokenizer.encode(data[0][i])[-hparams.max_seq_length:],
                #    'output':tokenizer.encode(data[1][i])[-hparams.max_seq_length:], 'task':all_tasks_list.index('response_generation')})
            elif (file!='all_prompts.json') and (file!='all_tasks.json'): # 'reading_comprehension.json'
                for task in all_tasks:
                    if file.replace('.json','') in all_tasks[task]: 
                        break
                grounding = data[0]
                query = data[1]
                answers = data[2]
                print(f"task: {task}, dataset:{file}, cases:{len(query)}")
                for i in range(len(query)):
                    prompts = all_prompts[file.replace('1','').replace('0','').replace('.json','')]
                    t = random.sample(prompts,1)[0]#random.choice(prompts,k=2)
                    if isinstance(t, list):
                        if task in ['response_generation', 'intent', 'dst', 'kg'] or file=='race.json': # no new prompts are added here
                            seq =  (grounding[i] if grounding != [] else '') + ' ' + query[i]
                        elif task =='nlg' or task=='summary': # old prompts need to be removed in those two categories
                            if task == 'nlg':
                                ps = random.sample(generation_prompts,1)
                            elif task == 'summary':
                                ps = random.sample(summary_prompts,1)
                            for p in ps:
                                if isinstance(p,list):
                                    seq = p[0] + ' '+ (grounding[i] if grounding != [] else query[i]) + ' ' + p[1]
                                else:
                                    seq =  (grounding[i] if grounding != [] else query[i]) + ' ' + p
                        else:
                            if task == 'emotional': # file.replace('.json','') in all_tasks['emotional']:
                                ps = random.sample(emotion_prompts,1)
                            #elif task == 'summary': # file.replace('.json','') in all_tasks['summary']:
                            #    ps = random.sample(summary_prompts,1)
                            elif task in ['doc_qa','dialqa']:
                                ps = random.sample(answer_prompts,2)
                            elif task =='text2sql':
                                ps = random.sample(txt2sql_prompts,1)
                            elif task =='multiple_choice':
                                ps = random.sample(choice_prompts,1)
                            # elif task =='nlg': # old prompts need to be removed
                            #    ps = random.sample(generation_prompts,1)
                            for p in ps:
                                if isinstance(p,list):
                                    seq = p[0] + ' '+ (grounding[i] if grounding != [] else '') + ' ' + query[i] + ' ' + p[1]
                                else:
                                    seq =  (grounding[i] if grounding != [] else '') + ' ' + query[i] + ' ' + p #(grounding[i] if grounding!=[] else query[i])

                        #else:
                        #    if grounding != []:
                        #        if file in ['dailydialog0.json']:
                        #            seq = grounding[i]  + ' ' + query[i] 
                        #        else:
                        #            seq = t[0] + ' '+ grounding[i] + ' '+t[1] + ' '+ query[i] + ' '+ (t[2].replace('.','') if t[2].replace('.','').replace(':','') not in answers[i] else '')
                        #    else:
                        #        seq = t[0] + ' '+ query[i] + ' '+ t[1]
                    else:
                        if grounding != []:
                            if 'task description ahead' in t:
                                seq = ' Task description: ' +t.replace('task description ahead,','') + (grounding[i] if grounding != [] else '')  + ' ' + query[i] 
                            elif 'task description' in t:
                                seq = (grounding[i] if grounding != [] else '') + ' ' + query[i] + ' ' +t.replace('task description,','') 
                            else:
                                seq = (grounding[i] if grounding != [] else '') + ' ' + query[i] 
                    """
                    else:
                        if 'task description ahead' in t:
                            seq.append(t.split(',')[1] + grounding[i] + ' ' + query[i] + ' ')
                        elif 'task description' in t:
                            seq.append(grounding[i] + ' '  + t.split(',')[1]+ query[i] + ' ')
                        else:
                            seq.append(grounding[i] + ' '  + t.split(',')[1]+ query[i] + ' ' + t.replace('_',' '))
                    """
                    if i < 10:
                        self.debug.append({'input':seq,'output':answers[i], 'task': all_tasks_list.index(task)})
                    #self.dataset.append({'input':tokenizer.encode(seq)[-hparams.max_seq_length:],'output':tokenizer.encode(answers[i])[-hparams.max_seq_length:], 'task': all_tasks_list.index(task)})
                    self.dataset.append({'input':seq[-hparams.max_seq_length:],
                    'output':answers[i][-hparams.max_seq_length:], 'task': all_tasks_list.index(task)})
        num = len(self.dataset)
        random.shuffle(self.dataset)
        #with open(self.train_file, "w", encoding="utf-8") as fp:
        #for ex in self.dataset[int(num/20):]:
        #fp.write(json.dumps(ex) + "\n")
        with open(self.train_file, "w", encoding="utf-8") as fp:
            json.dump(self.dataset, fp)
            # json.dump(self.dataset[int(num/20):], fp)
        with open(self.eval_file, "w", encoding="utf-8") as fp1:
            json.dump(self.dataset[:int(num/20)], fp1)
        #with open(self.eval_file, "w", encoding="utf-8") as fp1:
        #for ex1 in self.dataset[:int(num/20)]:
        #    fp1.write(json.dumps(ex1) + "\n")
        json.dump(self.debug,open('data/pre_train/data_overview.json','w'),indent=2)
        #json.dump(dataset,open(path,'w'))
        return  

def get_unsup(tokenizer, hparams):
    dir = 'data/pre_train/prompts'
    all_tasks = json.loads(open(os.path.join(dir,'all_tasks.json'), 'r').read())
    all_tasks_list = [k for k in all_tasks]
    data = json.loads(open(os.path.join(dir,'space_un.json'), 'r').read())
    dataset = []
    debug = []
    for i in tqdm(range(len(data[0]))): # use 1/10 unannoted dialogs
        dataset.append({'input':tokenizer.encode(data[0][i])[-hparams.max_seq_length:],
        'output':tokenizer.encode(data[1][i])[-hparams.max_seq_length:], 'task':all_tasks_list.index('response_generation')})
        if i <=10 or (i==100000):
            debug.append({'input':data[0][i],'output':data[1][i], 'task':all_tasks_list.index('response_generation')})
    random.shuffle(dataset)
    with open('data/pre_train/unsup_encoded.json','w') as fp:
        json.dump(dataset, fp)
    print(1)
    json.dump(debug, open('data/pre_train/unsup_overview.json','w'),indent=2)

def merge_data():
    proportion = 50 # 
    labeled_path = 'pre_train/train_encoded_all.json'
    eval_path = 'pre_train/train_encoded_all.json'
    unlabeled_path = 'pre_train/unsup_encoded.json'
    final_train_path = f"pre_train/train_encoded_{proportion}.jsonl"
    final_eval_path = f"pre_train/eval_encoded_{proportion}.jsonl"
    unlabeled_data = json.load(open(unlabeled_path, 'r'))
    train_data = json.load(open(labeled_path, 'r'))
    unsup_num = int( len(unlabeled_data) * proportion / 100 )
    train_data.extend(unlabeled_data[:unsup_num])
    eval_data = json.load(open(eval_path, 'r'))
    with open(final_train_path, "w", encoding="utf-8") as fp:
        for ex in train_data:
            fp.write(json.dumps(ex) + "\n")
    with open(final_eval_path, "w", encoding="utf-8") as fp:
        for ex1 in eval_data:
            fp.write(json.dumps(ex1) + "\n")
    return
    
if __name__ == "__main__":
    import argparse
    from transformers import T5Tokenizer
    from args import parse_args
    parser = argparse.ArgumentParser()
    hparams = parse_args(parser)
    hparams.model_name = 't5-base'
    hparams.max_seq_length = 512
    hparams.type = 'train'
    tokenizer = T5Tokenizer.from_pretrained(hparams.model_name)
    #get_unsup(tokenizer, hparams)
    loader = LazyDataset(tokenizer,hparams)
    #LazyDataset.load_dataset(tokenizer,hparams)
    # use merge data to get final datasets
    #merge_data()
