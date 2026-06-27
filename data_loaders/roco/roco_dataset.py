import os
import json
import torch
import copy
from torch.utils.data import Dataset
from PIL import Image
from transformers import PreTrainedTokenizer
import random

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


class VQADataset(Dataset):
    def __init__(self, data_path, tokenizer: PreTrainedTokenizer, image_folder, transform=None, max_seq_len=512, split="train", phrase_type=None, question_type=None):
        if data_path.endswith(".txt"):
            self.data = []
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(None, 1)  # splits on any whitespace (tab or space), only once
                    if len(parts) != 2:
                        print(f"[WARN] Could not parse line: {line}")
                        continue
                    image_name, caption = parts
                    self.data.append({
                        "image_name": image_name.strip() + ".jpg",  # e.g., ROCO_68327.jpg
                        "caption": caption.strip()
                    })
            # self.data = []
            # with open(data_path, "r") as f:
            #     for line in f:
            #         line = line.strip()
            #         if not line:
            #             continue
            #         if '\t' in line:
            #             image_name, caption = line.split('\t', 1)
            #         else:
            #             image_name, caption = line.split(' ', 1)  # fallback
            #         self.data.append({"image_name": image_name.strip() + ".jpg", "caption": caption.strip()})
        else:
            # Load data
            with open(data_path, 'r') as f:
                self.data = json.load(f)

            # Apply filtering based on split type
            if split == "train":
                self.data = [item for item in self.data if item["phrase_type"] in ["freeform", "para"] and "test" not in item["phrase_type"]]
            elif split == "test":
                self.data = [item for item in self.data if "test" in item["phrase_type"]]

            # Additional filtering
            if phrase_type is not None:
                self.data = [item for item in self.data if item["phrase_type"] == phrase_type]

            if question_type is not None:
                self.data = [item for item in self.data if item["question_type"] == question_type]

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

        # Extract question & answer
        question = random.choice(self.instruction_pool)
        answer = data_item.get("caption", "")
        if not isinstance(answer, str):
            answer = str(answer)

        image_name = data_item.get("image_name", None)
        # if image_name:
        #     image_path = os.path.join(self.image_folder, image_name)
        #     try:
        #         image = Image.open(image_path).convert("RGB")  # Use PIL for image loading
        #     except Exception as e:
        #         print(f"Warning: Could not load image {image_path}. Error: {e}")
        #     if self.transform:
        #         image = self.transform(image)
        # else:
        #     image = torch.zeros(3, 224, 224)
        if image_name:
            image_path = os.path.join(self.image_folder, image_name)
            try:
                image = Image.open(image_path).convert("RGB")  # Use PIL for image loading
            except Exception as e:
                print(f"Warning: Could not load image {image_path}. Error: {e}")
                image = Image.new("RGB", (224, 224), (0, 0, 0))  # Fallback black image
                # image = torch.zeros(3, 224, 224)

            if self.transform:
                image = self.transform(image)
        else:
            print('missing_roco')
            image = torch.zeros(3, 224, 224)

        # Format prompt
        formatted_prompt = format_prompt(question)
        full_input = formatted_prompt + " " + answer

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
            "answers": answer,
            "image_name": image_name
        }