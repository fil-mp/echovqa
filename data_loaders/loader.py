import os
from torch.utils.data import DataLoader, ConcatDataset

from data_loaders.roco.roco_dataset import VQADataset as RocoDataset
from data_loaders.cxr_dataset1 import MedicalLazySupervisedDataset as CXRDataset
from data_loaders.medicat.medicat_dataset import Medicat as MedicatDataset
from data_loaders.imageclef.imageclef_data import ImageClefDataset

from models.tokenizer import Tokenizer

# ---- Pretraining corpus paths (edit to your local copies) ----
# Obtain ROCO, MIMIC-CXR, MedICaT, and ImageCLEF from their original sources.
ROCO_BASE        = "/path/to/roco-dataset/data"          # contains {train,val,test}/{radiology,non-radiology}/
CXR_JSON_DIR     = "/path/to/cxr"                        # contains {train,val,test}.json
CXR_IMAGE_FOLDER = "/path/to/mimic/resized"
MEDICAT_JSON     = "/path/to/medicat/subcaptions_public.json"
MEDICAT_IMAGES   = "/path/to/medicat/figures"
IMAGECLEF_CSV    = "/path/to/imageclef/prompt-gt.csv"
IMAGECLEF_IMAGES = "/path/to/imageclef/images"


def get_combined_dataloader(
    batch_size=1,
    shuffle=True,
    max_seq_len=512,
    transform=None,
    hf_model_path="/path/to/llama",   # dir containing tokenizer.model
):
    tokenizer = Tokenizer(model_path=os.path.join(hf_model_path, "tokenizer.model"))

    roco_splits = []
    for split in ["train", "val", "test"]:
        for domain in ["radiology", "non-radiology"]:
            cap_path = f"{ROCO_BASE}/{split}/{domain}/captions.txt"
            img_path = f"{ROCO_BASE}/{split}/{domain}/images"
            if os.path.exists(cap_path) and os.path.exists(img_path):
                roco_splits.append(RocoDataset(
                    data_path=cap_path,
                    tokenizer=tokenizer,
                    image_folder=img_path,
                    transform=transform,
                    max_seq_len=max_seq_len,
                    split=split,
                ))

    cxr_splits = []
    for split in ["train", "val", "test"]:
        data_path = os.path.join(CXR_JSON_DIR, f"{split}.json")
        if os.path.exists(data_path):
            cxr_splits.append(CXRDataset(
                data_path=data_path,
                tokenizer=tokenizer,
                image_folder=CXR_IMAGE_FOLDER,
                transform=transform,
                split=split,
            ))

    medicat_dataset = MedicatDataset(
        data_path=MEDICAT_JSON,
        tokenizer=tokenizer,
        image_folder=MEDICAT_IMAGES,
        transform=transform,
        max_seq_len=max_seq_len,
        split="train",
    )

    imageclef_dataset = ImageClefDataset(
        data_path=IMAGECLEF_CSV,
        tokenizer=tokenizer,
        image_folder=IMAGECLEF_IMAGES,
        transform=transform,
        max_seq_len=max_seq_len,
        split="train",
    )

    combined_dataset = ConcatDataset(
        roco_splits + cxr_splits + [medicat_dataset, imageclef_dataset]
    )
    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=8, pin_memory=True)