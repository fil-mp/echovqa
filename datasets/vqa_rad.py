import os
import json
import torch
import copy
import random
from torch.utils.data import Dataset
from PIL import Image
from transformers import PreTrainedTokenizer

IGNORE_INDEX = -100  # Mask for non-target tokens


def format_prompt(instruction, input=None):
    PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        ),
    }
    if input is None:
        return PROMPT_DICT["prompt_no_input"].format_map({'instruction': instruction})
    return PROMPT_DICT["prompt_input"].format_map({'instruction': instruction, 'input': input})


class VQADataset(Dataset):
    def __init__(self, data_path, tokenizer: PreTrainedTokenizer, image_folder,
                 transform=None, max_seq_len=512, split="train",
                 phrase_type=None, question_type=None):
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.image_folder = image_folder
        self.transform = transform
        self.max_seq_len = max_seq_len

        print(f"Loaded {len(self.data)} samples for {split} split.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]

        question = data_item["question"]
        answer = data_item.get("answer", "")
        if not isinstance(answer, str):
            answer = str(answer)

        image_name = data_item.get("image_name", None)
        if image_name:
            image_path = os.path.join(self.image_folder, image_name)
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Warning: could not load image {image_path}. Error: {e}")
                image = Image.new("RGB", (224, 224), (0, 0, 0))  # fallback so transform never sees an undefined image
            if self.transform:
                image = self.transform(image)
        else:
            image = torch.zeros(3, 224, 224)

        formatted_prompt = format_prompt(question)
        full_input = formatted_prompt + " " + answer

        input1 = torch.tensor(self.tokenizer.encode(formatted_prompt, bos=True, eos=False), dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer.encode(full_input, bos=True, eos=True), dtype=torch.int64)

        padding = self.max_seq_len - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:self.max_seq_len]

        labels = copy.deepcopy(input2)
        labels[:len(input1)] = IGNORE_INDEX

        input_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input_mask] = 0
        labels[~label_mask] = 0
        input_mask = input_mask.float()

        return {
            "input_ids": input2,
            "labels": labels,
            "attention_mask": input_mask,
            "pixel_values": image,
            "questions": question,
            "answers": answer,
            "answer_type": data_item.get("answer_type", "OPEN"),
        }
