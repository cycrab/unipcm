def convert_sample_to_shot_dialKG(sample,with_knowledge=True): # need modification to fit the pretraining style
    prefix = "Dialogue:\n"
    assert len(sample["dialogue"]) == len(sample["meta"])
    for turn, meta in zip(sample["dialogue"],sample["meta"]):
        prefix += f"user: {turn[0]}" +"\n"
        if with_knowledge and len(meta)>0:
            prefix += 'KG: '
            for m in meta:
                if isinstance(m,list):
                    prefix += ' '.join(m[0]) + ', '
            prefix = prefix[:-2]
            prefix += "\n"
        if turn[1] == "":
            prefix += f"system:" 
            return prefix
        else:
            prefix += f"system: {turn[1]}" +"\n"
            
    return prefix


def convert_sample_to_shot_dialKG_interact(sample,with_knowledge):
    prefix = "Dialogue:\n"
    assert len(sample["dialogue"]) == len(sample["KG"])
    for turn, meta in zip(sample["dialogue"],sample["KG"]):
        prefix += f"user: {turn[0]}" +"\n"
        if with_knowledge and len(meta)>0:
            prefix += f"KG: {meta[0]}" +"\n"
        if turn[1] == "":
            prefix += f"system:" 
            return prefix
        else:
            prefix += f"system: {turn[1]}" +"\n"
            
    return prefix
