def convert_sample_to_shot_coQA(sample, with_knowledge=None):
    prefix = f"{sample['meta']}\n"

    for turn in sample["dialogue"]:
        prefix += f" Question: {turn[0]}" +"\n" # Q:
        if turn[1] == "":
            prefix += f" Answer:" # A:
            return prefix
        else:
            prefix += f" Answer: {turn[1]}" +"\n" # A:

    return prefix


