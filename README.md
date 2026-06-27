# EchoVQA — Method Code

Official implementation for "EchoVQA: Enabling Conversational Assistance for Point-of-Care Cardiac Ultrasound"

Training and evaluation code: a LLaMA-Adapter VLM with a
BiomedCLIP vision encoder, visual-query adapters, and deep prompts, for
multi-turn cardiac-ultrasound VQA. The pipeline has two stages: pretraining on
public medical image-text corpora, then fine-tuning on EchoVQA dataset.

## Install

```bash
pip install -r requirements.txt
```

You also need:
- **LLaMA-2-7B (chat) weights** in a single directory containing `tokenizer.model`
  and the HuggingFace model files. Pass that directory via `--llm-model-path`.

## Datasets

**EchoVQA** (this work): https://huggingface.co/datasets/filbel/echoVQA

The released EchoVQA is in LLaVA conversation format. Convert it to the flat VQA
format used by the fine-tuning loader with:

```bash
python prepare_vqa.py --in-dir /path/to/echoVQA --out-dir /path/to/echoVQA/vqa
```

This produces `{train,test,val}_vqa.json` with `{question, answer, image_name, answer_type}`
records, which `finetune.sh` consumes.

**Pretraining corpora** (obtain each from its source under the respective terms):
[ROCO](https://link.springer.com/chapter/10.1007/978-3-030-01364-6_20),
[MIMIC-CXR](https://physionet.org/content/mimic-cxr/),
[MEDICAT](https://arxiv.org/abs/2010.06000),
[ImageCLEF](https://www.imageclef.org/).
Point to them in `data_loaders/loader.py`.

## Fine-tuning on EchoVQA

Edit the paths at the top of `finetune.sh`, then:

```bash
bash finetune.sh
```

It initializes from a pretrained checkpoint (`--resume-from`) and trains on the
EchoVQA train json, evaluating on the test json every few epochs.

## Pretraining

Edit `pretrain.sh` and run:

```bash
bash pretrain.sh
```