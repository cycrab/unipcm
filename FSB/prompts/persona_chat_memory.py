def convert_sample_to_shot_msc(sample, with_knowledge=None):
    prefix = "User info:\n"
    for s in sample["meta"]["user"]:
        prefix += s+"\n"

    prefix += "System info:\n"
    for s in sample["meta"]["assistant"]:
        prefix += s+"\n"

    prefix += "Dialogue:\n"
    for turn in sample["dialogue"]:
        prefix += f"user: {turn[0]}" +"\n"
        if turn[1] == "":
            prefix += f"system:" 
            return prefix
        else:
            prefix += f"system: {turn[1]}" +"\n"

    return prefix


def convert_sample_to_shot_msc_interact(sample, with_knowledge=None):
    prefix = "User Information:\n"
    for s in sample["user"]:
        prefix += s+"\n"

    prefix += "Assistant Information:\n"
    for s in sample["assistant"]:
        prefix += s+"\n"

    prefix += "Dialogue:\n"
    for turn in sample["dialogue"]:
        prefix += f"user: {turn[0]}" +"\n"
        if turn[1] == "":
            prefix += f"system:" 
            return prefix
        else:
            prefix += f"system: {turn[1]}" +"\n"

    return prefix
