# Unipcm (prompt aware instruction-tuning for dialog tasks)
<!--This repository contains code and data for the **SIGIR'2022** paper "**UniPCM: Universal Pre-trained Conversation Model with Task-based Automatic Prompt**".
-->

## 1. Main Results
SPACE-X performs end-to-end dialog modeling, dialog understanding, open-domain chit-chat, which achieves new state-of-the-art results on benchmark datasets including: MultiWOZ2.0(full data), and DialoGLUE(few-shot)，and improves performance over strong baseline on chit-chat datasets Persona and DailyDialog, detailed results can be found in the paper.

### Task 1：Dialog understanding

Codes can be found in Folder dialoglue Folder(3.7)(The implementation of MultiWOZ2.1-DST is in Folder DST-as-Prompting(3.6)).

### Task 2：Chit-chat

<!--对话状态跟踪是要预测多个键值对是否正确，因此指标是联合准确率 Joint Goal Accuracy，评判每轮所有键值对正确才算对。-->
Codes can be found in Folder FSB(3.5)

### Task 3：End-to-End Dialog Modeling
Result:
| Dataset Name | Inform | Success |  BLEU | Combined Score |
|:-------------------:|:------:|:-------:|:-----:|:--------------:|
|     MultiWOZ2.0     |  88.30 |  76.80  | 19.20 |     101.80     |

Codes can be found in Folder MTTOD(3.4)

<!--**NOTE**: Inform指标用于评估多轮对话的理解能力，Success指标用于评估多轮对话的任务完成率，BLEU指标用于评估每轮系统生成回复语句的流畅度，Combined Score = (Inform + Success) * 0.5 + BLEU。

一个完整的多轮对话流程：在每轮对话交互中，系统首先理解用户的Query行为，生成对话状态 Belief State(又称 Dialog State)，用于查询数据库，得到领域相关的查询结果，然后生成系统动作System Act，用于指导最终回复语句的生成。-->


## 2. Requirements
```
- torch == 1.7.1
- numpy == 1.18.5
- nltk == 3.5
- spacy == 2.3.5
- tqdm == 4.60.0
- transformers == 4.18.0
```
We use the tokenization tool in SpaCy and you can directly install python packages by commands: `pip install -r requirements.txt` and `python -m spacy download en_core_web_sm`.

## 3. Scripts
### 3.1. Python scripts
#### 3.1.1 Pretraining
Our pretrained model can be download from this [link](http://spacex-prompted.oss-cn-hangzhou-zmf.aliyuncs.com/prompts/unipcm-full.zip)(few-shot model can be found in [link](http://spacex-prompted.oss-cn-hangzhou-zmf.aliyuncs.com/prompts/unipcm-few.zip))

The pretraining datasets are on this [link](http://spacex-prompted.oss-cn-hangzhou-zmf.aliyuncs.com/prompts) with eval sets (eval_encoded_all.json), train sets (train_encoded_all.json), unsupervised data(unsup_encoded.json).
dataset.py is used for mixing data for final pretraining, run this script to get the final datasets.

get_data.py is used for collecting pretraining data, mainly from huggingface.dataset. 
generate_template.py is used for automatically generating prompts. 
train.py is used for pretraining. 

In summary, to pretrain the model, you should :
First, download pretraining corpus through ```wget http://spacex-prompted.oss-cn-hangzhou-zmf.aliyuncs.com/prompts/{train ,eval, unsup}```.
Second, mixing data using ```python dataset.py```, unlabeled proportion can be set in 
the function merge_data().
Third, run ```python -m torch.distributed.launch --nproc_per_node=8 train.py``` to train the model, it will take about 72h on 8 80G A100 cards. If your gpu do not have enough memory space, please change the batch-size.

#### 3.1.2 Promt generation
generate_template.py is used for automatically generating prompts.
prompt_visualization.py is used for visualizing prompts. 

To generate prompts for new task, you should :
run ```python generate_template.py```
First, use function generate_template_using_keyword(task) to generate prompts. Write your own function to get_{task} and add the function to the dict variable get_domain_data to get training instances for prompt generation. Moreover keywords should be added to keyword_dict.
Second, use function get_final(task) to get filtered prompts.
Third, use function post_process() to get final prompts.

### 3.2. Folder extra data
This folder contains datasets for pretraining, collected from various github repos.
The original github repos link is in the README.md file in the folder, and the dataset can be downloaded through git clone command.

### 3.3. Folder data
Contains some useful data.

### 3.4. Folder MTTOD
The folder contains our experiment in Multi-woz END2END task.
The details of the original github repos and our modification are in the README.md file in the folder.

### 3.5. Folder FSB
The folder contains our experiment in chit-chat datasets Persona and DailyDialog.
The details of the original github repos and our modification are in the README.md file in the folder.

### 3.6. Folder DST-as-Prompting
The folder contains our experiment in Multi-woz DST task.
The details of the original github repos and our modification are in the README.md file in the folder.

### 3.7. Folder dialoGLUE
The folder contains our experiment in DialoGLUE benchmark.
The details of the original github repos and our modification are in the README.md file in the folder.

### 3.8. natural-instructions-2.7
The folder contains the original datasets of Supernatural Instruction, which are used for extracting keywords from task instructions and generating relevant prompts based on task instances.
The details of the original github repos and our modification are in the README.md file in the folder.

<!--
```
SPACE/
├── data  # multiwoz2.0 and banking77 datasets
├── db  # database in multiwoz2.0 dataset
├── model  # bert vocabulary
├── outputs  # fine-tuned checkpoints for multiwoz2.0 and banking77
├── scripts  # inference bashes for multiwoz2.0 and banking77
├── space  # model and modules
├── tools  # misc tools
└── trippy  # separated code for dialog state tracking (dst)
    ├── data  # multiwoz2.2 datasets
    ├── dataset_config  # data configuration
    ├── model  # dst model and modules
    ├── outputs  # fine-tuned checkpoints for multiwoz2.2
    └── scripts  # # inference bashes for multiwoz2.2
```
-->
