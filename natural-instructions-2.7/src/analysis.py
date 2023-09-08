import argparse
import glob
import random
import json 
import os

from gensim.models import doc2vec
from gensim import corpora,models,similarities
import nltk # for english tokenization
from transformers import BertTokenizer
from transformers import BertModel
from torch.nn import CosineSimilarity
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from tqdm import tqdm
#from stanfordcorenlp import StanfordCoreNLP

def get_key_word():
    threshold = 0.5
    prompts = json.load(open('task_description.json', 'r'))
    task_prompts = {}
    key_words = {}
    texts = []
    for task, ps in prompts.items():
        for p in ps:
            if task not in task_prompts:
                task_prompts[task] = []
                key_words[task] = []
            task_prompts[task].append(nltk.word_tokenize(p))
            texts.append(nltk.word_tokenize(p))
    dictionary = corpora.Dictionary(texts)
    feature_cnt=len(dictionary.token2id)
    corpus=[dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(corpus)
    index=similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=feature_cnt)
    #utt_num=len(corpus)
    count = -1
    for t, ps in task_prompts.items():
        for p in ps:
            count = count + 1
            for word_count in range(len(p)):
                words = [p[word_count:word_count+1], # 1gram, 2gram and 3gram
                p[word_count:word_count+2], p[word_count:word_count+3]]
                for word in words:
                    sen = dictionary.doc2bow(word)
                    sen_tfidf = tfidf[sen]
                    sim=index[sen_tfidf][count]
                    if sim > threshold:
                        if word not in key_words[t]:
                            key_words[t].append(word)
    json.dump(key_words, open('keywords.json', 'w'), indent=2)
    return

def get_key_word_bert():
    device = 7
    wanted_pos = ['VBN', 'NN', 'VB', 'JJ', 'NNP', 'VBG', 'NNS', 'NNPS']
    prompts = json.load(open('task_description.json', 'r'))
    task_prompts = {}
    texts = []
    for task, ps in prompts.items():
        for p in ps:
            if task not in task_prompts:
                task_prompts[task] = []
            processed = []
            for pos in pos_tag(word_tokenize(p)):
                if pos[1] in wanted_pos:
                    processed.append(pos[0])
            task_prompts[task].append(processed)

    threshold = 0.85
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    model = BertModel.from_pretrained('bert-large-uncased')
    model.to(device)
    key_words_final = {}
    for k, ps in tqdm(task_prompts.items()):
        key_words_final[k] = []
        t = tokenizer(k, return_tensors="pt").to(device)
        embed_k = model(**t)[0].mean(dim=1) # shape 1024
        for p in ps:
            for word_count in range(len(p)):
                words = [p[word_count:word_count+1], # 1gram, 2gram and 3gram
                p[word_count:word_count+2], p[word_count:word_count+3]]
                for word in words:
                    tmp = tokenizer(' '.join(word), return_tensors="pt").to(device)
                    embed_p = model(**tmp)[0].mean(dim=1)
                    cos_sim = CosineSimilarity(dim=1)
                    sim = cos_sim(embed_k, embed_p)
                    k_word = ' '.join(word)
                    if sim> threshold and (k_word not in key_words_final[k]) and k_word not in k:
                        key_words_final[k].append(k_word)
    json.dump(key_words_final, open('keywords_final_bert_0.9.json', 'w'), indent=2)
    return

def postprocess():
    key_words_final = []
    keywords = json.load(open('keywords_final_bert_0.9.json', 'r'))
    for k, prompts in keywords.items():
        key_words_final[k] = []
        for p in prompts:
            if ' '.join(p) not in k and ' '.join(p) not in key_words_final[k]:
                key_words_final[k].append(' '.join(p))
    json.dump(key_words_final, open('keywords_final_bert.json', 'w'), indent=2)

    return
def analysis_key_word():
    threshold = 0.75
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    model = BertModel.from_pretrained('bert-large-uncased')
    key_words = json.load(open('keywords.json', 'r'))
    key_words_final = {}
    for k, ps in key_words.items():
        key_words_final[k] = []
        t = tokenizer(k, return_tensors="pt")
        embed_k = model(**t)[0].mean(dim=1) # shape 1024
        for p in ps:
            tmp = tokenizer(' '.join(p), return_tensors="pt")
            embed_p = model(**tmp)[0].mean(dim=1)
            cos_sim = CosineSimilarity(dim=1)
            sim = cos_sim(embed_k, embed_p)
            if sim> threshold:
                key_words_final[k].append(p)
    json.dump(key_words_final, open('keywords_final.json', 'w'), indent=2)
    return

def analysis():
    count = 0 # total English task: 1037
    cats = {}
    cats_description = {}
    domains = {}
    sources = {}
    st_file = []
    wanted_cats = ['Fill in The Blank', 'Question Understanding', 'Question Answering', 'Named Entity Recognition',
    'Text Categorization', 'Commonsense Classification', 'Dialogue Generation', 'Data to Text', 'Summarization',
    'Keyword Tagging', 'Dialogue State Tracking']
    for file in os.listdir('tasks'):
        if file!='README.md':
            dataset = json.load(open(os.path.join('tasks',file),'r'))
            if (dataset["Input_language"] == ["English"]) and (dataset["Output_language"] == ["English"]) and(
                dataset["Instruction_language"] == ["English"]): # and (dataset['Categories'][0] in wanted_cats) : 
                count =count + 1
                for cat in dataset["Categories"]:
                    if cat not in cats:
                        cats[cat] = [] 
                        cats_description[cat] = []
                    #cats[cat] = ca
                    cats[cat].append(dataset) # ['Instances']
                    cats_description[cat].append(dataset['Definition'][0])
                for domain in dataset["Domains"]:
                    if domain not in domains:
                        domains[domain] = 0
                    domains[domain] = domains[domain] + 1
                for source in dataset["Source"]:
                    if source not in sources:
                        sources[source] = 0
                    sources[source] = sources[source] + 1
                
            else:
                st_file.append(dataset)
    json.dump(cats_description, open('task_description.json', 'w'), indent=2)
    return

def get_task_data():
    count = 0 # total English task: 1037
    cats = {}
    cats_instance = {}
    domains = {}
    sources = {}
    st_file = []
    wanted_cats = ['Fill in The Blank', 'Question Understanding', 'Question Answering', 'Named Entity Recognition',
    'Text Categorization', 'Commonsense Classification', 'Dialogue Generation', 'Data to Text', 'Summarization',
    'Keyword Tagging', 'Dialogue State Tracking']
    for file in os.listdir('tasks'):
        if file!='README.md':
            dataset = json.load(open(os.path.join('tasks',file),'r'))
            if (dataset["Input_language"] == ["English"]) and (dataset["Output_language"] == ["English"]) and(
                dataset["Instruction_language"] == ["English"]): # and (dataset['Categories'][0] in wanted_cats) : 
                count =count + 1
                for cat in dataset["Categories"]:
                    if cat not in cats:
                        cats[cat] = [] 
                        cats_instance[cat] = []
                    #cats[cat] = ca
                    #cats[cat].append(dataset) # ['Instances']
                    cats_instance[cat].extend(dataset['Instances'][:500])
                for domain in dataset["Domains"]:
                    if domain not in domains:
                        domains[domain] = 0
                    domains[domain] = domains[domain] + 1
                for source in dataset["Source"]:
                    if source not in sources:
                        sources[source] = 0
                    sources[source] = sources[source] + 1
                
            else:
                st_file.append(dataset)
    json.dump(cats_instance, open('task_instances.json', 'w'), indent=2)
    return

if __name__ == '__main__':
    #analysis()
    # get_key_word()
    #analysis_key_word()
    #get_key_word_bert()
    #postprocess()
    get_task_data()