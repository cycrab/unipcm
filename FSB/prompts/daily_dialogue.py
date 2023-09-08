def convert_sample_to_shot_DD_prefix(sample, with_knowledge=None):
    prefix = "Dialogue:\n"
    for turn in sample["dialogue"]:
        prefix += f"user: {turn[0]}" +"\n"
        if turn[1] != "" :
            prefix += f"system: {turn[1]}" +"\n"

    return prefix

def convert_sample_to_shot_DD_inference(sample, with_knowledge=None, gpt=False):
    if gpt:
        prefix = ""
        for turn in sample["dialogue"]:
            if turn[0] != "":
                prefix += f"{turn[0]}" +"\n"
            else:
                return prefix
            if turn[1] != "" :
                prefix += f"{turn[1]}" +"\n"
            else:
                return prefix
    else:
        #prefix = "Dialogue:\n"
        prefix = ""
        for turn in sample["dialogue"]:
            if turn[0] != "":
                prefix += f"user: {turn[0]}" +" \n "
            else:
                prefix += f"user:"
                return prefix

            if turn[1] != "" :
                prefix += f"system: {turn[1]}" +" \n "
            else:
                prefix += f"system:"
                return prefix

    return prefix
