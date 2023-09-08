#  Few-Shot Bot: Prompt-Based Learning for Dialogue Systems(modified)

Our work modifies upon the original code on that we add code for fintuning in ```main_response_generation.py```. Moreover, to increase speed in the inference time, we add batch generation for testing.

To run our model on the scripts, we made some modification for T5 model, which can be adjusted through parameter 'gpt=False'

To replicate our results in the paper, run ```main_response_generation.py``` using different "model, model_type, datasets, labeled porportion", which can be set in the parameters.

# Origin Readme:
This repository includes the dataset, experiments results, and code for the paper:

**Few-Shot Bot: Prompt-Based Learning for Dialogue Systems** [PDF](https://arxiv.org/pdf/2110.08118.pdf). 

**Authors**: [Andrea Madotto](https://andreamad8.github.io), [Zhaojiang Lin](https://zlinao.github.io), [Genta Indra Winata](https://gentawinata.com/), [Pascale Fung](https://pascale.home.ece.ust.hk/)

## Installation
In this repo, we load all the validation and test sets used in the evaluation. For running the experiments and the demo, you should install the following requirements:
```
pip install -r requirements.txt
```
## Modification

some datasets are not used, therefore to save space, some origin files under generations and data folder are removed and can be found by cloning the original repos.

## Basic Running

### Reproducing the results and plots
The ```generation``` folder stores the generated responses of the experiments in all datasets. To generate the tables and the plots in the paper, run:
```
python generate_plots_tables.py
```
This script loads all the files computes the mean between different runs and generates the plots. Note that this script is very custom for each dataset, but it can serve as a guideline for future extensions. 


### Running the experiments
There are three main files to run 1) response generation (```main_response_generation.py```), 2) conversational parsing (```main_conversational_parsing.py```), and 3) skill-selector (```main_skill_selector.py```). In these files, we load the necessary prompt (```load_prefix```), and we run the generation (```generate_response```) for each sample in the test set. Since each dialogue skill require a different template, as shown in the paper, we create a function that converts structured data into the correct shot prompt. An example of this function can be found in ```prompts/persona_chat.py```, and in ```generic_prompts.py``` we store the generation functions. 

In each main file, there is a configuration object (```mapper```) that specifies meta-information about the task (i.e., number of shots, generation length, decoding type, prompt converter). Especially for conversational parsing, there are different decoding types. For example, in MWOZ, the model generates the dialogue state, which is further looped into the next turn. 


#### How to run?
For example, to run the persona chat experiments (0, 1, k-shots), you can use the following command:
```
python main_response_generation.py --model_checkpoint EleutherAI/gpt-j-6B --dataset persona --gpu 0
```
In case your GPU has less that 16GB, then you could add ```--multigpu``` to spawn 4 GPUs (e.g., 1080Ti) and do inference in parallel. Similarly, for conversational parsing tasks, you could use:
```
python main_conversational_parsing.py --model_checkpoint EleutherAI/gpt-j-6B --dataset wow-parse --gpu 0
```
Notice that some parsing task requires a knowledge base (e.g., dialKG-parse requires the KG in neo4j). 
Finally, to run the skill-selector task, you could use:
```
python main_skill_selector.py --model_checkpoint EleutherAI/gpt-j-6B --shots_k 6 --repetition 1 --gpu 0
```
where repetition is the seed for selecting random samples in the prompts. 

#### Runners
In the ```runners``` folder, we provide a rudimental runner to run all the experiments and reproduce the results in the paper. 

## Few-Shot Bot
There are two FSB modes: 1) controlled style generation (FSB-CG) and 2) full-model. 

### FSB-CG 
Check the ```FSB-CG.ipynb``` to try to interact with FSB in your local machine, or try directly in colab at 
```
https://colab.research.google.com/drive/15hQv1V3Cs5kQVfLOE_FZc1VCWQ3YpWVd?usp=sharing
```
Remember to select the environment with GPU!! 

### FSB 
Check the ```FSB.ipynb``` to try to interact with FSB in your local machine, or try directly in colab at 
```
https://colab.research.google.com/drive/1JkPeX-6oXikiwWKqW5QkibEy8Oq6KM9g?usp=sharing
```
Remember to select the environment with GPU!! This current version does not query the Internet, Wiki and KGs, but only parse the dialogue history with MSC-parse. We implement only 4 skills for now. 

## Safety Bench
We benchmark the FSB using the safety bench provided in [ParlAI](https://github.com/facebookresearch/ParlAI/tree/main/projects/safety_bench). We implemented the [FSB wrapper](https://github.com/andreamad8/FSB/blob/main/utils/fsb_wrapper.py) and copy it in the ```model_wrappers``` folder (this require to pull the FSB repo in the ParlAI folder). Then, we run the benchmark: 
```
python projects/safety_bench/run_unit_tests.py -w fsb_wrapper --log-folder /tmp/fsb6B
```
We implemented a safety layer by adding three safety skills ([safety_topic](https://github.com/andreamad8/FSB/tree/main/data/safety_layers/sensitive_topics), [safety_nonadv](https://github.com/andreamad8/FSB/tree/main/data/safety_layers/human_nonadv_safety_eval), [safety_adv](https://github.com/andreamad8/FSB/tree/main/data/safety_layers/bot_adversarial_dialogue_datasets_with_persona)). If the safety skill is triggered, we use a predefined response (```Shall we talk about something else?```).

The results from the FSB (6B), with and without the safety layer, are shown in Table 12 and 13, and the full-report at:
```
https://github.com/andreamad8/FSB/tree/main/data/safety_layers/results_FSB_6B_withsafetyskills
https://github.com/andreamad8/FSB/tree/main/data/safety_layers/results_FSB_6B_withoutsafetyskills
```
