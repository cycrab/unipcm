import os
import sys
import json
from glob import glob
from random import sample
import copy

from clean_dataset import clean_slot_values

domain_desc_flag = True # To append domain descriptions or not 
slot_desc_flag = True  # To append slot descriptions or not 
PVs_flag = True # for categorical slots, append possible values as suffix
def get_few():
  all_ids = []
  few = json.load(open('multiwoz_few/train_dials.json','r'))
  for id,dial in few.items():
    all_ids.append(id)
  json.dump(all_ids, open('few_ids.json','w'))
  return all_ids

def get_eval_test():
  eval_file = 'multiwoz/data/MultiWOZ_2.1/valListFile.txt'
  test_file = 'multiwoz/data/MultiWOZ_2.1/testListFile.txt'
  eval_list = []
  test_list = []
  tmp = '1'
  with open(eval_file, "r", encoding="utf-8") as fp:
    while tmp != "":
      tmp = fp.readline()
      if tmp != '':
          eval_list.append(tmp.replace('\n','')) # .replace('','.json')
    fp.close()
  tmp = '1'
  with open(test_file, "r", encoding="utf-8") as fp1:
    while tmp != "":
        tmp = fp1.readline()
        if tmp != '':
          test_list.append(tmp.replace('\n',''))
    fp1.close()
  return eval_list, test_list

def preprocess(dial_json, schema, out, idx_out, excluded_domains, frame_idxs, out_few=None, idout_few=None, mode=None):
  if isinstance(dial_json, dict):
    new_dial_json = []
    for id, dial in dial_json.items():
      new_dial = {}
      new_dial['turns'] = dial['log']
      new_dial['dialogue_id'] = id
      new_dial_json.append([])
  else:
    dial_json_n = dial_json.split("/")[-1]
    dial_json = open(dial_json)
    dial_json = json.load(dial_json)
  few_id = get_few()
  eval_id, test_id = get_eval_test()
  for dial_idx in range(len(dial_json)):
    dial = dial_json[dial_idx]
    cur_dial = ""
    for turn in dial["turns"]:
      speaker = " [" + turn["speaker"] + "] " 
      uttr = turn["utterance"]
      cur_dial += speaker
      cur_dial += uttr  

      if turn["speaker"] == "USER":
        active_slot_values = {}
        for frame_idx in range(len(turn["frames"])):
          frame = turn["frames"][frame_idx]
          for key, values in frame["state"]["slot_values"].items():
            value = sample(values,1)[0]
            active_slot_values[key] = value

        # iterate thourgh each domain-slot pair in each user turn 
        for domain in schema:
          # skip domains that are not in the testing set
          if domain["service_name"] in excluded_domains:
            continue
          slots = domain["slots"]
          for slot in slots:
            d_name, s_name = slot["name"].split("-")
            # generate schema prompt w/ or w/o natural langauge descriptions
            schema_prompt = ""
            schema_prompt += " [domain] " + d_name + " " + domain["description"] if domain_desc_flag else d_name
            schema_prompt += " [slot] " + s_name + " " + slot["description"] if slot_desc_flag  else s_name
            if PVs_flag:
              # only append possible values if the slot is categorical
              if slot["is_categorical"]:
                PVs = ", ".join(slot["possible_values"])
                schema_prompt += " [PVs] " + PVs

            if slot["name"] in active_slot_values.keys():
              target_value = active_slot_values[slot["name"]]
            else:
              # special token for non-active slots
              target_value = "NONE"
            
            line = { "dialogue": cur_dial + schema_prompt, "state":  target_value }
            out.write(json.dumps(line))
            out.write("\n")
            if out_few and (dial['dialogue_id'] in few_id):
              out_few.write(json.dumps(line))
              out_few.write("\n")

            # write idx file for post-processing deocding
            idx_list = [ dial_json_n, str(dial_idx), turn["turn_id"], str(frame_idxs[d_name]), d_name, s_name ]
            idx_out.write("|||".join(idx_list))
            idx_out.write("\n")
            if idout_few and (dial['dialogue_id'] in few_id):
              idx_out.write("|||".join(idx_list))
              idx_out.write("\n")
  return

def preprocess_21(dial_json, schema, out, idx_out, excluded_domains, frame_idxs, out_few=None, idout_few=None, mode=None):
  dial_domains = [d['service_name'] for d in schema]
  dial_domains.remove('bus')
  all_slots = []
  for d in schema:
    for s in d['slots']:
      all_slots.append(s['name'])
  few_id = get_few()
  eval_id, test_id = get_eval_test()
  special = []
  for id, dial in dial_json.items():
    cur_dial = ""
    for turn in dial["log"]:
      if turn['metadata'] == {}:
        system = True
      else:
        system = False
      speaker = " system: " if system else " user: "
      uttr = turn["text"]
      cur_dial += speaker
      cur_dial += uttr  

      if not system:
        active_slot_values = {}
        dial_state = turn["metadata"]
        for domain in dial_domains:
          # inplementation from damd
          info_sv = dial_state[domain]['semi']
          for s,v in info_sv.items():
            s,v = clean_slot_values(domain, s,v)
            # if len(v.split())>1:
            #    v = ' '.join([token.text for token in self.nlp(v)]).strip()
            if v != '':
                active_slot_values[(domain + '-' + s)] = v
          book_sv = dial_state[domain]['book']
          for s,v in book_sv.items():
            if s == 'booked':
                continue
            s,v = clean_slot_values(domain, s,v)
            # if len(v.split())>1:
            #    v = ' '.join([token.text for token in self.nlp(v)]).strip()
            if v != '':
              tmp_slot = domain + '-book' + s
              if tmp_slot not in all_slots:
                active_slot_values[ (domain + '-' + s) ] = v
                if tmp_slot not in special:
                  special.append(tmp_slot)
              else:
                active_slot_values[tmp_slot] = v
          # for key, values in sv.items():
          #  value = sample(values,1)[0]
          #  active_slot_values[key] = value

        # iterate thourgh each domain-slot pair in each user turn 
        for domain in schema:
          # skip domains that are not in the testing set
          if domain["service_name"] in excluded_domains:
            continue
          slots = domain["slots"]
          for slot in slots:
            d_name, s_name = slot["name"].split("-")
            # generate schema prompt w/ or w/o natural langauge descriptions
            schema_prompt = ""
            schema_prompt += " [domain] " + d_name + " " + domain["description"] if domain_desc_flag else d_name
            schema_prompt += " [slot] " + s_name + " " + slot["description"] if slot_desc_flag  else s_name
            if PVs_flag:
              # only append possible values if the slot is categorical
              if slot["is_categorical"]:
                PVs = ", ".join(slot["possible_values"])
                schema_prompt += " [PVs] " + PVs

            if slot["name"] in active_slot_values.keys():
              target_value = active_slot_values[slot["name"]]
            else:
              # special token for non-active slots
              target_value = "NONE"
            
            line = { "dialogue": cur_dial + schema_prompt, "state":  target_value }
            idx_list = [ mode, id, str(dial['log'].index(turn)), str(frame_idxs[d_name]), d_name, s_name ]
            
            if mode=='train':
              if (id not in test_id) and (id not in eval_id):
                out.write(json.dumps(line))
                out.write("\n")
                idx_out.write("|||".join(idx_list))
                idx_out.write("\n")
                if out_few and (id in few_id):
                  out_few.write(json.dumps(line))
                  out_few.write("\n")
                  idout_few.write("|||".join(idx_list))
                  idout_few.write("\n")
            elif mode=='dev':
              if id in eval_id:
                out.write(json.dumps(line))
                out.write("\n")
                idx_out.write("|||".join(idx_list))
                idx_out.write("\n")
            elif mode=='test':
              if id in test_id:
                out.write(json.dumps(line))
                out.write("\n")
                idx_out.write("|||".join(idx_list))
                idx_out.write("\n")
  return

def main():
    data_path = "multiwoz/data/MultiWOZ_2.2/"
    # data_path = sys.argv[1]

    schema_path = data_path + "schema.json"
    schema = json.load(open(schema_path))
    frame_idxs = {"train": 0, "taxi":1, "bus":2, "police":3, "hotel":4, "restaurant":5, "attraction":6, "hospital":7}

    # skip domains that are not in the testing set
    excluded_domains = ["police", "hospital", "bus"]
    for split in ["train", "dev", "test"]:
        print("--------Preprocessing {} set---------".format(split))
        out = open(os.path.join(data_path, "{}.json".format(split)), "w")
        idx_out = open(os.path.join(data_path, "{}.idx".format(split)), "w")
        if split=='train':
          out_few = open(os.path.join(data_path, "{}_few.json".format(split)), "w")
          idx_outfew = open(os.path.join(data_path, "{}_few.idx".format(split)), "w")
        else:
          out_few = None
          idx_outfew = None
        dial_jsons = glob(os.path.join(data_path, "{}/*json".format(split)))
        for dial_json in dial_jsons:
            if dial_json.split("/")[-1] != "schema.json":
                preprocess(dial_json, schema, out, idx_out, excluded_domains, frame_idxs, out_few, idx_outfew)
        if idx_out:
          idx_out.close()
        if out_few:
          out_few.close()
        if idx_outfew:  
          idx_outfew.close()
        if out:
          out.close()
    print("--------Finish Preprocessing---------")

def processing_21(data_path):
  schema_path = data_path + "schema.json"
  schema = json.load(open(schema_path, 'r'))
  frame_idxs = {"train": 0, "taxi":1, "bus":2, "police":3, "hotel":4, "restaurant":5, "attraction":6, "hospital":7}

  # skip domains that are not in the testing set
  excluded_domains = ["police", "hospital", "bus"]
  for split in ["train", "dev", "test"]:
      print("--------Preprocessing {} set---------".format(split))
      out = open(os.path.join(data_path, "{}.json".format(split)), "w")
      idx_out = open(os.path.join(data_path, "{}.idx".format(split)), "w")
      if split=='train':
        out_few = open(os.path.join(data_path, "{}_few.json".format(split)), "w")
        idx_outfew = open(os.path.join(data_path, "{}_few.idx".format(split)), "w")
      else:
        out_few = None
        idx_outfew = None
      dial_jsons = json.load(open(os.path.join(data_path, "data.json"), 'r'))
      #for dial_json in dial_jsons:
      #    if dial_json.split("/")[-1] != "schema.json":
      preprocess_21(dial_jsons, schema, out, idx_out, excluded_domains, frame_idxs, out_few, idx_outfew, mode=split)
      if idx_out:
        idx_out.close()
      if out_few:
        out_few.close()
      if idx_outfew:  
        idx_outfew.close()
      if out:
        out.close()
  print("--------Finish Preprocessing---------")
  return
if __name__=='__main__':
    #main() # processing multiwoz2.2
    data_path = "multiwoz/data/MultiWOZ_2.1/"
    processing_21(data_path) 
    # this is not usable due to different data format between multiwoz2.1 and multiwoz2.2
    # use preprocessing from ubar or pptod instead
    # get_few()
