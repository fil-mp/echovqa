import os
import json
import torch
import copy
from torch.utils.data import Dataset
from PIL import Image
from transformers import PreTrainedTokenizer
import random
import pandas as pd

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
    return PROMPT_DICT["prompt_no_input"].format_map({'instruction': instruction}) if input is None else PROMPT_DICT["prompt_input"].format_map({'instruction': instruction, 'input': input})


class ImageClefDataset(Dataset):
    def __init__(self, data_path, tokenizer: PreTrainedTokenizer, image_folder, transform=None, max_seq_len=512, split="train", phrase_type=None, question_type=None):
        self.data = pd.read_csv(data_path, sep=';', names=["Prompt", "Filename"], header=0).to_dict(orient="records")

        self.tokenizer = tokenizer
        self.image_folder = image_folder
        self.transform = transform
        self.max_seq_len = max_seq_len
        self.instruction_pool = [
            'Briefly describe this image.',
            'Provide a concise depiction of this image.',
            'Present a short description of this image.',
            'Summarize this image in a few words.',
            'A short image caption:',
            'A short image description:',
            'A photo of ',
            'An image that shows ',
            'Write a short description for the image.',
            'Write a description for the photo.',
            'Provide a description of what is presented in the photo.',
            'Briefly describe the content of the image.',
            'Can you briefly explain what you see in the image?',
            'Could you use a few words to describe what you perceive in the photo?',
            'Please provide a short depiction of the picture.',
            'Using language, provide a short account of the image.',
            'Use a few words to illustrate what is happening in the picture.',
        ]

        print(f"Loaded {len(self.data)} samples for {split} split.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        raw_prompt = data_item["Prompt"]
        image_name = data_item["Filename"]

        # Extract question & answer
        for prefix in ["Generate an image", "Generate a"]:
            if raw_prompt.lower().startswith(prefix.lower()):
                caption = raw_prompt[len(prefix):].strip(" .")
                break
        else:
            caption = raw_prompt.strip(" .")
        question = random.choice(self.instruction_pool)

        # image_name = data_item.get("image_name", None)
        if image_name:
            image_path = os.path.join(self.image_folder, image_name)
            try:
                image = Image.open(image_path).convert("RGB")  # Use PIL for image loading
            except Exception as e:
                print(f"Warning: Could not load image {image_path}. Error: {e}")
            if self.transform:
                image = self.transform(image)
        else:
            print('missing_imageclef')
            image = torch.zeros(3, 224, 224)

        # Format prompt
        formatted_prompt = format_prompt(question)
        full_input = formatted_prompt + " " + caption

        # Tokenization
        input1 = torch.tensor(self.tokenizer.encode(formatted_prompt, bos=True, eos=False), dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer.encode(full_input, bos=True, eos=True), dtype=torch.int64)

        # Padding / Truncation
        padding = self.max_seq_len - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:self.max_seq_len]

        # Mask the question tokens
        labels = copy.deepcopy(input2)
        labels[:len(input1)] = IGNORE_INDEX  # Mask instruction tokens

        # Convert padding to zeros
        input_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input_mask] = 0
        labels[~label_mask] = 0
        input_mask = input_mask.float()
        label_mask = label_mask.float()


        return {
            "input_ids": input2,  # Tokenized full input
            "labels": labels,  # Masked labels
            "attention_mask": input_mask,  # Attention mask
            "pixel_values": image,  # Image tensor
            "questions": question,
            "answers": caption,
            "image_name": image_name
        }