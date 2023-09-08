def convert_sample_to_shot_persona(sample, with_knowledge=None, gpt=False):
    if gpt:
        #prefix = "system information:\n"
        prefix = "system information:\n "
        for s in sample["meta"]:
            prefix += s+" \n "
        prefix += "Dialogue:\n "
        for turn in sample["dialogue"]:
            prefix += f"{turn[0]}" +" \n "
            if turn[1] == "":
                return prefix
            else:
                prefix += f"{turn[1]}" +" \n " 
    else:
        prefix = "system information:\n "
        for s in sample["meta"]:
            prefix += s+" \n "

        prefix += "Dialogue:\n "
        for turn in sample["dialogue"]:
            prefix += f"user: {turn[0]}" +" \n "
            if turn[1] == "":
                prefix += f"system:" 
                return prefix
            else:
                prefix += f"system: {turn[1]}" +" \n "

    return prefix


