from datasets import load_dataset
dataset = load_dataset("HuggingFaceTB/smoltalk")

def convert_to_chatml(example):
    return {
        "messages": [
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
    }