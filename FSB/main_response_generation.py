import json
import os
import argparse
import numpy as np
import random

from prompts.generic_prompt import load_prefix, load_prefix_by_category, evalute_ppl, generate_response, load_samples, finetune
from prompts.persona_chat import convert_sample_to_shot_persona
from prompts.bAbi_dialogue import convert_sample_to_shot_bAbi
from prompts.coQA import convert_sample_to_shot_coQA
from prompts.persona_chat_memory import convert_sample_to_shot_msc
from prompts.wizard_of_wikipedia import convert_sample_to_shot_wow
from prompts.wizard_of_internet import convert_sample_to_shot_wit
from prompts.emphatetic_dialogue import convert_sample_to_shot_ed
from prompts.dialKG import convert_sample_to_shot_dialKG
from prompts.daily_dialogue import convert_sample_to_shot_DD_prefix, convert_sample_to_shot_DD_inference
from prompts.image_chat import convert_sample_to_shot_IC_prefix, convert_sample_to_shot_IC_inference
from prompts.image_chat_with_img import convert_sample_to_shot_IC_img_prefix, convert_sample_to_shot_IC_img_inference
from prompts.smd import convert_sample_to_shot_smd, convert_sample_to_shot_smd_custum
from prompts.semantic_parser import convert_sample_to_shot_semantic_parser
from tabulate import tabulate
from metric.scorer import score
from collections import defaultdict
import os
from tqdm import tqdm
import glob
from utils.utils import load_model, save_file, checker_file

os.environ["TOKENIZERS_PARALLELISM"] = "false"

mapper = {
          "persona": {"shot_converter":convert_sample_to_shot_persona, 
                    "shot_converter_inference": convert_sample_to_shot_persona,
                     "file_data":"data/persona/","with_knowledge":None,
                     "shots":6,"shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":3
                     ,"batch-size":16}, # with persona info
        "smd-navigate": {"shot_converter":convert_sample_to_shot_smd_custum, 
                    "shot_converter_inference": convert_sample_to_shot_smd,
                     "file_data":"data/smd/navigate-","with_knowledge":None,
                     "shots":8,"shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":5},
        "smd-schedule": {"shot_converter":convert_sample_to_shot_smd, 
                    "shot_converter_inference": convert_sample_to_shot_smd,
                     "file_data":"data/smd/schedule-","with_knowledge":None,
                     "shots":8,"shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":5},
        "smd-weather": {"shot_converter":convert_sample_to_shot_smd_custum, # 8
                    "shot_converter_inference": convert_sample_to_shot_smd,
                     "file_data":"data/smd/weather-","with_knowledge":None,
                     "shots":8,"shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":5},
          "msc-dialogue-2": {"shot_converter":convert_sample_to_shot_msc, 
                    "shot_converter_inference": convert_sample_to_shot_msc,
                     "file_data":"data/msc/session-2-","with_knowledge":None,
                     "shots":3,"shot_separator":"\n\n", # 3
                     "meta_type":"all","gen_len":50,"max_number_turns":3,
                     "batch_size":8},
          "wow": {"shot_converter":convert_sample_to_shot_wow, 
                 "shot_converter_inference": convert_sample_to_shot_wow,
                 "file_data":"data/wow/","with_knowledge":True,
                  "shots":1,"shot_separator":"\n\n",
                  "meta_type":"incremental","gen_len":50,"max_number_turns":5}, # 1
          "wit": {"shot_converter":convert_sample_to_shot_wit, 
                 "shot_converter_inference": convert_sample_to_shot_wit,
                 "file_data":"data/wit/","with_knowledge":True,
                  "shots":2,"shot_separator":"\n\n",
                  "meta_type":"incremental","gen_len":60,"max_number_turns":4}, # 2
          "ed": {"shot_converter":convert_sample_to_shot_ed, 
                 "shot_converter_inference": convert_sample_to_shot_ed,
                 "file_data":"data/ed/","with_knowledge":None,
                  "shots":17,"shot_separator":"\n\n",
                  "meta_type":"none","gen_len":50,"max_number_turns":5}, #17
          "dialKG": {"shot_converter":convert_sample_to_shot_dialKG, 
                 "shot_converter_inference": convert_sample_to_shot_dialKG,
                 "file_data":"data/dialKG/","with_knowledge":True,
                  "shots":9,"shot_separator":"\n\n",
                  "meta_type":"incremental","gen_len":50,"max_number_turns":4}, #9
          "DD": {"shot_converter":convert_sample_to_shot_DD_inference, #convert_sample_to_shot_DD_prefix,  # 
                 "shot_converter_inference": convert_sample_to_shot_DD_inference,
                 "file_data":"data/dailydialog/","with_knowledge":False,
                  "shots":6,"shot_separator":"\n\n",
                  "meta_type":"all","gen_len":50,"max_number_turns":5}, # 6 # all_turns
          "IC": {"shot_converter":convert_sample_to_shot_IC_prefix, 
                 "shot_converter_inference": convert_sample_to_shot_IC_inference,
                 "file_data":"data/image_chat/","with_knowledge":False,
                  "shots":{1024:[0,1,5],2048:[0,1,10]},"shot_separator":"\n\n",
                  "meta_type":"all_turns_category","gen_len":50,"max_number_turns":5},
          "IC-img": {"shot_converter":convert_sample_to_shot_IC_img_prefix, 
                 "shot_converter_inference": convert_sample_to_shot_IC_img_inference,
                 "file_data":"data/image_chat/img_","with_knowledge":False,
                  "shots":{1024:[0,1,4],2048:[0,1,10]},"shot_separator":"\n\n",
                  "meta_type":"all_turns","gen_len":50,"max_number_turns":5},
          "coQA": {"shot_converter":convert_sample_to_shot_coQA, 
                    "shot_converter_inference": convert_sample_to_shot_coQA,
                     "file_data":"data/coQA/","with_knowledge":None,
                     "shots":{1024:[0],2048:[0,1]},"shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":5},
          "babi5-first": {"shot_converter":convert_sample_to_shot_bAbi, 
                    "shot_converter_inference": convert_sample_to_shot_bAbi,
                     "file_data":"data/dialog-bAbI-tasks/bAbI-dial-5-first-","with_knowledge":None,
                     "shots":{2048:[8]},"shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":10},
          "babi5-first-OOV": {"shot_converter":convert_sample_to_shot_bAbi, 
                    "shot_converter_inference": convert_sample_to_shot_bAbi,
                     "file_data":"data/dialog-bAbI-tasks/bAbI-dial-5-OOV-first-","with_knowledge":None,
                     "shots":{2048:[8]},"shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":10},
          "babi5-second": {"shot_converter":convert_sample_to_shot_bAbi, 
                    "shot_converter_inference": convert_sample_to_shot_bAbi,
                     "file_data":"data/dialog-bAbI-tasks/bAbI-dial-5-second-","with_knowledge":None,
                     "shots":{2048:[2]},"shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":10},
          "babi5-second-OOV": {"shot_converter":convert_sample_to_shot_bAbi, 
                    "shot_converter_inference": convert_sample_to_shot_bAbi,
                     "file_data":"data/dialog-bAbI-tasks/bAbI-dial-5-OOV-second-","with_knowledge":None,
                     "shots":{2048:[2]},"shot_separator":"\n\n",
                     "meta_type":"all","gen_len":50,"max_number_turns":10},
         }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", default="dialogpt",type=str) # ,required=True
    parser.add_argument("--gpt", default=True, type=bool) 
    # set gpt=True if the model is gpt-based, for example, dialogpt
    # models/t5-pretrain, models/t5-pptod (pptod), t5-base, dialogpt, blenderbot
    # gpt2, t5-large, models/t5-pretrained
    
    parser.add_argument("--dataset", default="persona",type=str) # ,required=True
    #"persona", "smd-navigate", "smd-schedule", "smd-weather", "msc-dialogue-2", "wow", "wit", "ed", "dialKG", "DD", "IC", skip "IC-image"
    # "coQA", "babi5-first", "babi5-first-OOV"
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--sample_times", type=int, default=2)
    parser.add_argument("--do_sample", action='store_true', help="sample n times and rescore based on ppl")
    parser.add_argument("--multigpu", action='store_true', help="run on multiple gpus")
    parser.add_argument("--verbose", action='store_true', help="run on multiple gpus")
    parser.add_argument("--finetune", default=True, help="wether to finetune the model or not")
    parser.add_argument("--epoch_num", default=3, help="the epoch_num to finetune")
    parser.add_argument("--lr", default=1e-5, type=float, help="learning rate")
    parser.add_argument("--proportion", default=10, type=int, help="proportion of data")
    parser.add_argument("--seed", default=8, type=int, help="random seed")
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' # 'cpu'
    random.seed(args.seed)
    beam = args.beam
    model_checkpoint = args.model_checkpoint

    model, tokenizer, max_seq = load_model(args,model_checkpoint,device)
    #gpt2_model, gpt2_tokenizer, max_seq_gpt = load_model(args,'gpt2',device)

    # dialogpt
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # import torch
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    # model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

    # facebook/blenderbot-400M-distill 
    # from transformers.models.blenderbot import BlenderbotTokenizer, BlenderbotForConditionalGeneration
    # tokenizer = BlenderbotTokenizer.from_pretrained('facebook/blenderbot-400M-distill')
    # model = BlenderbotForConditionalGeneration.from_pretrained('facebook/blenderbot-400M-distill')
    # model_output = model.generate(input_ids, num_beams=1, do_sample=True, top_p=0.9, num_return_sequences=5, return_dict=False)
    
    list_of_dataset = args.dataset.split(",")
    for d in list_of_dataset:
        print(f"EVALUATING DATASET {d} on {model_checkpoint} with beam size {beam}")
        if "category" in mapper[d]['meta_type']: # not use
            """
            prefix_list = [{0:''}]
            prefix_list = load_prefix_by_category(tokenizer=tokenizer, shots_value=mapper[d]["shots"][max_seq], 
                                shot_converter=mapper[d]["shot_converter"], 
                                file_shot=mapper[d]["file_data"]+"valid.json", 
                                name_dataset=d, with_knowledge=mapper[d]["with_knowledge"], 
                                shot_separator=mapper[d]["shot_separator"],sample_times=args.sample_times)if 't5' not in args.model_checkpoint else [{0:''}] # override prefix loading

            for id_prefix, prefix_shot_by_category in enumerate(prefix_list):
                shot_results = defaultdict(lambda: defaultdict(list))
                for cat, prefix_shots in tqdm(prefix_shot_by_category.items()):

                    for shots, prefix in prefix_shots.items():
            """
            if checker_file(f"{d}_{shots}_{model_checkpoint}_{beam}-{args.do_sample}_{id_prefix}.json") or args.verbose:
                shot_results[shots]["generated_out"] += generate_response(model, tokenizer, shot_converter=mapper[d]["shot_converter_inference"], 
                                                file_to_eval=mapper[d]["file_data"]+f"test/{cat}.json", prefix=prefix, 
                                                device=device, max_number_turns=mapper[d]["max_number_turns"], 
                                                with_knowledge=mapper[d]["with_knowledge"], 
                                                meta_type=mapper[d]["meta_type"], gen_len=mapper[d]["gen_len"], 
                                                beam=beam, max_seq=max_seq, eos_token_id=198, 
                                                do_sample=args.do_sample, multigpu=args.multigpu)

                ppl_score = evalute_ppl(model, tokenizer, shot_converter=mapper[d]["shot_converter_inference"], 
                                        file_to_eval=mapper[d]["file_data"]+f"test/{cat}.json", 
                                        prefix=prefix, device=device, max_number_turns=mapper[d]["max_number_turns"], 
                                        with_knowledge=mapper[d]["with_knowledge"], max_seq=max_seq,
                                        meta_type=mapper[d]["meta_type"])
                shot_results[shots]["ppl"].append(ppl_score)
                        
                for shots, results in shot_results.items():
                    res_score = score(files_test=mapper[d]["file_data"]+f"test.json",files_to_score=results["generated_out"], meta_type="last_turn")
                    res_score["ppl"] = np.mean(results["ppl"])
                    print(res_score)
                    if checker_file(f"{d}_{shots}_{model_checkpoint}_{beam}-{args.do_sample}_{id_prefix}.json"):
                        save_file(f"{d}_{shots}_{model_checkpoint}_{beam}-{args.do_sample}_{id_prefix}.json", {"score":res_score,"generation":results["generated_out"]})
        else:
            """
            prefix_list = load_prefix(tokenizer=tokenizer, shots_value=mapper[d]["shots"][max_seq], 
                                    shot_converter=mapper[d]["shot_converter"], 
                                    file_shot=mapper[d]["file_data"]+"valid.json", 
                                    name_dataset=d, with_knowledge=mapper[d]["with_knowledge"], 
                                    shot_separator=mapper[d]["shot_separator"],sample_times=args.sample_times) if 't5' not in args.model_checkpoint else [{0:''}] # can be used to load prompt latter
        
            first_time = True
            for id_prefix, prefix_shots in enumerate(prefix_list):
               for shots, prefix in prefix_shots.items():
                   if (shots == 0 or 't5' in args.model_checkpoint) and not first_time: continue 
                   first_time = False
            """
            filename = f"{args.dataset}/{d}_{args.proportion}_{model_checkpoint.replace('/','')}_{beam}-{args.do_sample}-{args.finetune}_{args.epoch_num}.json"
            # if args.finetune:
            #    filename = filename.replace('.json','50_ft.json')
            result = checker_file(filename)
            if (not result) or args.verbose:
                if args.finetune:
                    sample_list,eval_list = load_samples(shot_converter=mapper[d]["shot_converter"], 
                                        meta_type=mapper[d]["meta_type"], max_number_turns=mapper[d]["max_number_turns"], train_shot=mapper[d]["file_data"]+"train.json", file_shot=mapper[d]["file_data"]+"valid.json", label_proportion=args.proportion, gpt=args.gpt, name_dataset=d ) # sample_times, with_knowledge=mapper[d]["with_knowledge"]
                    random.shuffle(sample_list)
                    finetune(sample_list, eval_list, model, tokenizer, device, args.lr, args.gpt, args.epoch_num)
                generation_out = generate_response(model, tokenizer, shot_converter=mapper[d]["shot_converter_inference"], 
                                                file_to_eval=mapper[d]["file_data"]+"test.json", 
                                                device=device, max_number_turns=mapper[d]["max_number_turns"], 
                                                with_knowledge=mapper[d]["with_knowledge"], 
                                                meta_type=mapper[d]["meta_type"], gen_len=mapper[d]["gen_len"], 
                                                beam=beam, max_seq=max_seq, eos_token_id=198, 
                                                do_sample=args.do_sample, multigpu=args.multigpu, verbose=args.verbose, gpt=args.gpt) # prefix
                save_file(filename, {"generation":generation_out}) # "score":res_score,
            else:
                generation_out = result["generation"]
                """
                if args.model_checkpoint=='blenderbot':
                    from transformers import T5Tokenizer # there's some issue with the blenderbot tokenizer, a temporary solution
                    tokenizer1 = T5Tokenizer.from_pretrained('t5-base')
                    for num in range(len(generation_out[0]['dialogue'][1])):
                        utt = generation_out[0]['dialogue'][1][num]
                        generation_out[0]['dialogue'][1][num] = ' '.join(tokenizer1.tokenizer(utt))
                """
            res_score = score(files_test=mapper[d]["file_data"]+"test.json",files_to_score=generation_out, meta_type=mapper[d]["meta_type"], result=generation_out[0]['dialogue'], gpt=args.gpt) # isinstance(generation_out['dialogue'], tuple)
            ppl_score = evalute_ppl(model, tokenizer, shot_converter=mapper[d]["shot_converter_inference"], # prefix
                                    file_to_eval=mapper[d]["file_data"]+"test.json", 
                                    device=device, max_number_turns=mapper[d]["max_number_turns"], 
                                    with_knowledge=mapper[d]["with_knowledge"], max_seq=max_seq,
                                    meta_type=mapper[d]["meta_type"], verbose=args.verbose, gpt=args.gpt)
            res_score["ppl"] = ppl_score
            print(res_score)
                        
