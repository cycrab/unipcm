import json

from sklearn import manifold
from transformers import BertTokenizer, T5Tokenizer
from transformers import BertModel, T5EncoderModel
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, CrossEntropyLoss

class Task_space(torch.nn.Module):
    def __init__(self, tokenizer):
        super(Task_space, self).__init__()
        task_num = 9
        self.bert_model=BertModel.from_pretrained('bert-base-cased')#cfg.model_path
        self.classifier = nn.Linear(self.bert_model.config.hidden_size, task_num) 

    def forward(self,input_ids: torch.tensor,
                attention_mask: torch.tensor,
                label):
        hidden_states = self.bert_model(input_ids=input_ids,attention_mask = attention_mask)[0]
        pooled_output =  hidden_states[:, :].mean(dim=1)
        loss_fct = CrossEntropyLoss()
        logits = self.classifier(pooled_output)
        if label is not None:
            loss = loss_fct(logits.view(-1, self.num_labels), label.type(torch.long))
        else:
            loss = 0.0
        return loss, pooled_output

def get_intent():
    origin_intent = json.load(open('data/prompts/intent.json', 'r'))
    ablation_i= origin_intent[""]
    ablation_intents = []
    for s, i in ablation_i.items():
        if len(i)==3:
            ablation_intents.append((i[0] + ' user:',i[1]+i[2]))
        elif len(i)==2:
            ablation_intents.append((i[0] + ' user:',i[1]))
    json.dump(ablation_intents[:7] ,open('ablation_intent.json', 'w'), indent=2)
    return

def analysis_keywords():
    keywords = json.load(open('natural-instructions-2.7/task_description.json'))
    return

def prompt_visualize():
    nums = {1:'t5-base',2:'t5-pretrained'}
    titles = {1:'(a) T5-base model',2:'(b) Our pre-trained model'}
    for num, model in nums.items():
        t5_tokenizer = T5Tokenizer.from_pretrained(model) # 't5-pretrained'
        t5_model = T5EncoderModel.from_pretrained(model)
        #tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        #model = Task_space(tokenizer)
        dir = 'data/new_prompts/'
        all_task = ['QA', 'choice', 'emotion', 'generation',
        'intent', 'response', 'state', 'summary', 'txt2sql'] # 'answer', 
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:orange', 'tab:purple'] # 'w'
        keyword_dict ={'intent':['user intent', 'user intention', "user's purpose", "user's purpose", "user's intention", 'user purpose', ''], 
        'state': ['user state', "user's state", 'user info', 'user information', 'user status', ''], 
        'emotion':['emotion', 'sentiment', 'feeling', 'feels', ''], 
        'summary':['summary', 'sum up', 'main idea', 'outline', ''], 
        'QA':['Answer', 'A:', 'answer'], 
        'generation':['generate', 'Generate', 'expression', 'express', 'keywords', ''], 
        'response':['Response', 'reponse', 'respond', 'reply', ''],
        'choice':['Choose', 'choose', 'choice'],
        'txt2sql':['information', 'format', 'change', 'structured language', 'structure', '']}
        # {'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
        all_embeddings = []
        task_index = 0
        task_indexes = [0]
        t5_model.eval()
        with torch.no_grad():  
            for task in all_task:
                file_name = dir + task + '_final.json'
                prompts = json.load(open(file_name, 'r'))
                for p in prompts:
                    if isinstance(p, list):
                        if task in ['intent', 'state']:
                            prompt = p[1]
                        else:
                            prompt = p[0] + p[1] # can be p[0] + p[1]
                    else:
                        prompt = p
                    flag = 0
                    for keyword in keyword_dict[task]:
                        if keyword in prompt:
                            flag = 1
                            break
                    if flag==1:
                        t = t5_tokenizer(prompt, return_tensors="pt")
                        #embedding_p = model(t['input_ids'], t['attention_mask'], None)[1].detach().numpy().squeeze()
                        #t = tokenizer( prompt, return_tensors="pt")
                        #embedding_k = model(t['input_ids'], t['attention_mask'], None)[1].detach().numpy().squeeze()
                        embedding_p = t5_model(**t)[0].mean(dim=1).detach().numpy().squeeze() # shape 1024
                        all_embeddings.append(embedding_p)
                        task_index = task_index + 1
                task_indexes.append(task_index)

        #for p in all_prompts:
        # tsne for embedding
        tSNE = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=5)
        tsne = tSNE.fit_transform(np.array(all_embeddings))
        centers = []
        #for i in range(len(all_task)):
        #    prompt_num = task_indexes[i+1] - task_indexes[i]
        #    centers.append((sum(tsne[task_indexes[i]:task_indexes[i+1], 0])/prompt_num,sum(tsne[task_indexes[i]:task_indexes[i+1], 1])/prompt_num))
        threshold = 10.0
        plt.subplot(2,1,num)
        for i in range(len(all_task)):
        #    processed_x = []
        #    processed_y = []
        #    prompt_num = task_indexes[i+1] - task_indexes[i]
        #    for j in prompt_num

            plt.scatter(x=tsne[task_indexes[i]:task_indexes[i+1], 0], y=tsne[task_indexes[i]:task_indexes[i+1], 1], label=all_task[i], color=colors[i], s=10) 
        #plt.title('t-SNE visualization of prompt embeddings',fontsize =16)
        #plt.xlabel('X',fontsize =16) # FPR 
        #plt.ylabel('Y',fontsize =16) # score   
        #plt.legend(bbox_to_anchor=(1.5, 1),loc='upper right', fontsize =12) # bbox_to_anchor=(0, 0), 
        plt.xticks(fontsize =16)
        plt.yticks(fontsize =16)
        plt.tight_layout()
        plt.title(titles[num], fontsize=16) # position=[-20, -55] ,
    plt.legend(bbox_to_anchor=(1.5, 1.6),loc='upper right', fontsize =12) # bbox_to_anchor=(0, 0), 
    plt.show()         
    #plt.savefig('embedding_visualization.jpg')
    return

if __name__ == "__main__":
    #get_intent()
    #analysis_keywords()
    prompt_visualize()
