import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer 
from transformers.models.blenderbot import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import json
import os

def load_model(args,model_checkpoint,device):
    print(f"LOADING {model_checkpoint}")
    if "gpt-j"in model_checkpoint or "neo"in model_checkpoint:
        model = AutoModelForCausalLM.from_pretrained(model_checkpoint, low_cpu_mem_usage=True)
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        if args.multigpu:
            from parallelformers import parallelize
            parallelize(model, num_gpus=4, fp16=True, verbose='detail')
        else:
            model.half().to(device)
        max_seq = 2048

    # model_output = model.generate(input_ids, num_beams=1, do_sample=True, top_p=0.9, num_return_sequences=5, return_dict=False)
    elif model_checkpoint=="blenderbot":
        model = BlenderbotForConditionalGeneration.from_pretrained('facebook/blenderbot_small-90M')
        # blenderbot_small-90M
        # blenderbot-3B
        # 'facebook/blenderbot-400M-distill'
        tokenizer = BlenderbotTokenizer.from_pretrained('facebook/blenderbot_small-90M')
        #
        model.resize_token_embeddings(len(tokenizer))
        max_seq = 512

    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # import torch
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    # model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    elif model_checkpoint=="dialogpt": # use medium as it is the same size
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        tokenizer.pad_token = 'pad'
        max_seq = 1024

    elif 't5' in model_checkpoint:
        tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
        # tokenizer.bos_token = ":"
        # tokenizer.eos_token = "\n"
        model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
        max_seq = 512

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        tokenizer.bos_token = ":"
        tokenizer.eos_token = "\n"
        model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
        max_seq = 1024

    model.to(device)
    print("DONE LOADING")
    
    return model, tokenizer, max_seq


def save_file(filename, results):
    filename = filename.replace("EleutherAI/","")
    with open(f'generations/{filename}', 'w') as fp:
        json.dump(results, fp, indent=4)

def checker_file(filename):
    filename = filename.replace("EleutherAI/","")
    result = None
    if os.path.exists(f'generations/{filename}'):
        print(f"generations/{filename} already exists! ==> Skipping the file" )
        result = json.load(open(f'generations/{filename}','r'))
    return result # os.path.exists(f'generations/{filename}'),
