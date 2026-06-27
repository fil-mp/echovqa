import os
import json
import torch
import copy
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
    return PROMPT_DICT["prompt_no_input"].format_map({'instruction': instruction}) if input is None else PROMPT_DICT["prompt_input"].format_map({'instruction': instruction, 'input': input})


class MedicalLazySupervisedDataset(Dataset):
    def __init__(self, data_path, tokenizer: PreTrainedTokenizer, image_folder, transform=None, max_seq_len=512, split="train", phrase_type=None, question_type=None):
        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        # Apply filtering based on split type
        # Only apply filtering if phrase_type field exists
        if self.data and "phrase_type" in self.data[0]:
            if split == "train":
                self.data = [item for item in self.data if item.get("phrase_type") in ["freeform", "para"] and "test" not in item.get("phrase_type", "")]
            elif split == "test":
                self.data = [item for item in self.data if "test" in item.get("phrase_type", "")]


        # Additional filtering
        if phrase_type is not None:
            self.data = [item for item in self.data if item["phrase_type"] == phrase_type]

        if question_type is not None:
            self.data = [item for item in self.data if item["question_type"] == question_type]

        self.tokenizer = tokenizer
        self.image_folder = image_folder
        self.transform = transform
        self.max_seq_len = max_seq_len

        print(f"Loaded {len(self.data)} samples for {split} split.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]

        # Extract question & answer
        question = data_item["question"]
        answer = data_item.get("answer", "")
        if not isinstance(answer, str):
            answer = str(answer)

        image_name = data_item.get("image_path", None)
        if image_name:
            image_path = os.path.join(self.image_folder, image_name)
            try:
                image = Image.open(image_path).convert("RGB")  # Use PIL for image loading
            except Exception as e:
                print(f"Warning: Could not load image {image_path}. Error: {e}")
            if self.transform:
                image = self.transform(image)
        else:
            print('missing_cxr')
            image = torch.zeros(3, 224, 224)  # Default black image if missing

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

        # Return a dictionary compatible with your training loop
        # return {
        #     "input_ids": input2,  # Tokenized full input
        #     "labels": labels,  # Masked labels
        #     "attention_mask": input_mask,  # Attention mask
        #     "pixel_values": image,  # Image tensor
        # }
        return {
            "input_ids": input2,  # Tokenized full input
            "labels": labels,  # Masked labels
            "attention_mask": input_mask,  # Attention mask
            "pixel_values": image,  # Image tensor
            "questions": question,
            "answers": answer,
            "image_name": image_name
        }