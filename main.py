import os
import argparse
import torch
import json
import logging
import numpy as np
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs, DeepSpeedPlugin
from accelerate.utils import gather_object
import open_clip

import util.lr_sched as lr_sched
from models.tokenizer import Tokenizer
from models.model import LLamaAdapter
from data_loaders.loader import get_combined_dataloader
from datasets.vqa_rad import VQADataset

import collections
from nltk.translate.bleu_score import sentence_bleu
from tabulate import tabulate
from metrix import calculate_exactmatch, calculate_f1score, calculate_exactmatch_pefomed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


def evaluate_model_metrics_complete(args, model, val_loader, tokenizer, accelerator, max_samples=None):
    """Generate predictions across all processes, gather, and evaluate on the complete set."""
    model.eval()
    local_predictions = []
    accelerator.print("Starting evaluation - gathering all data across processes...")

    with torch.no_grad():
        for batch in val_loader:
            images = batch["pixel_values"].to(accelerator.device, dtype=torch.bfloat16)
            questions = batch["questions"]
            answers = batch["answers"]
            answer_types = batch.get("answer_type", ["OPEN"] * len(questions))

            for img, question_text, gt_answer, ans_type in zip(images, questions, answers, answer_types):
                formatted_prompt = format_prompt(question_text)
                prompt_ids = tokenizer.encode(formatted_prompt, bos=True, eos=False)
                prompt_tensor = torch.tensor([prompt_ids], device=accelerator.device)
                try:
                    output = model.generate(
                        img.unsqueeze(0), prompt_tensor,
                        max_gen_len=30, temperature=0, top_p=0.9,
                    )
                    pred_answer = output[0] if isinstance(output, list) else str(output)
                    local_predictions.append({
                        "question": question_text,
                        "gt_answer": gt_answer,
                        "pred_answer": pred_answer,
                        "answer_type": ans_type,
                    })
                except Exception as e:
                    accelerator.print(f"Error generating sample: {str(e)}")
                    continue

    all_predictions = gather_object(local_predictions)

    if accelerator.is_main_process:
        accelerator.print(f"Gathered {len(all_predictions)} total predictions")
        if max_samples and len(all_predictions) > max_samples:
            import random
            random.seed(42)
            all_predictions = random.sample(all_predictions, max_samples)
            accelerator.print(f"Sampled {max_samples} predictions for evaluation")
        metrics = calculate_metrics_complete(args, all_predictions, accelerator)
        print_evaluation_results_complete(all_predictions, metrics, accelerator)
        return metrics
    return {}


def calculate_metrics_complete(args, predictions, accelerator):
    """Calculate metrics from the complete prediction set."""
    if not predictions:
        accelerator.print("No predictions to evaluate!")
        return {}

    closed_scores = collections.defaultdict(list)
    bleu_scores = collections.defaultdict(list)
    exact_scores_closed = collections.defaultdict(list)
    exact_scores = collections.defaultdict(list)
    exact_scores_closed_pef = collections.defaultdict(list)
    exact_scores_pef = collections.defaultdict(list)
    f1_scores = collections.defaultdict(list)

    closed_count = 0
    for pred_item in predictions:
        gt_value = pred_item['gt_answer'].lower().strip()
        pred_value = pred_item['pred_answer'].lower().strip()

        if pred_item["answer_type"] == "CLOSED":
            closed_count += 1
            closed_scores['hit'].append(1 if pred_value == gt_value else 0)
            exact_scores_closed['hit'].append(calculate_exactmatch(pred_value, gt_value))
            exact_scores_closed_pef['hit'].append(calculate_exactmatch_pefomed(pred_value, gt_value))
        else:  # OPEN
            exact_scores['hit'].append(calculate_exactmatch(pred_value, gt_value))
            exact_scores_pef['hit'].append(calculate_exactmatch_pefomed(pred_value, gt_value))
            f1_score, precision, recall = calculate_f1score(pred_value, gt_value)
            f1_scores['f1'].append(f1_score)
            f1_scores['precision'].append(precision)
            f1_scores['recall'].append(recall)
            bleu_scores['bleu_score'].append(sentence_bleu([gt_value.split()], pred_value.split()))
            bleu_scores['bleu_score_1'].append(sentence_bleu([gt_value.split()], pred_value.split(), weights=(1, 0, 0, 0)))
            bleu_scores['bleu_score_2'].append(sentence_bleu([gt_value.split()], pred_value.split(), weights=(0, 1, 0, 0)))
            bleu_scores['bleu_score_3'].append(sentence_bleu([gt_value.split()], pred_value.split(), weights=(0, 0, 1, 0)))

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0

    metrics = {
        'exact_match_closed': avg(exact_scores_closed['hit']),
        'exact_match_open': avg(exact_scores['hit']),
        'exact_match_closed_pef': avg(exact_scores_closed_pef['hit']),
        'exact_match_open_pef': avg(exact_scores_pef['hit']),
        'f1_score': avg(f1_scores['f1']),
        'precision': avg(f1_scores['precision']),
        'recall': avg(f1_scores['recall']),
        'bleu_score': avg(bleu_scores['bleu_score']),
        'bleu_score_1': avg(bleu_scores['bleu_score_1']),
        'bleu_score_2': avg(bleu_scores['bleu_score_2']),
        'bleu_score_3': avg(bleu_scores['bleu_score_3']),
        'closed_accuracy': avg(closed_scores['hit']),
    }
    correct_closed = sum(closed_scores['hit'])
    correct_open = sum(exact_scores['hit'])
    total_closed = len(closed_scores['hit'])
    total_open = len(exact_scores['hit'])
    total = total_closed + total_open
    metrics.update({
        'num_closed': total_closed, 'num_open': total_open, 'num_total': total,
        'correct_closed': correct_closed, 'correct_open': correct_open,
        'average_accuracy': (correct_closed + correct_open) / total if total > 0 else 0.0,
    })
    accelerator.print(f"Evaluation completed on {total} samples ({total_closed} closed, {total_open} open)")
    return metrics


def print_evaluation_results_complete(predictions, metrics, accelerator):
    accelerator.print(f"\n=== Sample Predictions (showing {min(5, len(predictions))}) ===")
    for i in range(min(5, len(predictions))):
        pred = predictions[i]
        accelerator.print(f"Q: {pred['question']}")
        accelerator.print(f"GT: {pred['gt_answer']}")
        accelerator.print(f"Pred: {pred['pred_answer']}")
        accelerator.print(f"Type: {pred['answer_type']}")
        accelerator.print("-" * 50)

    results_table = tabulate(
        [
            ['Exact Match (Open)', metrics['exact_match_open'] * 100],
            ['Exact Match (Closed)', metrics['exact_match_closed'] * 100],
            ['Exact Match (Open, PeFoMed)', metrics['exact_match_open_pef'] * 100],
            ['Exact Match (Closed, PeFoMed)', metrics['exact_match_closed_pef'] * 100],
            ['Average Accuracy (All)', metrics['average_accuracy'] * 100],
            ['F1 Score', metrics['f1_score'] * 100],
            ['Precision', metrics['precision'] * 100],
            ['Recall', metrics['recall'] * 100],
            ['BLEU Score', metrics['bleu_score'] * 100],
            ['BLEU Score 1-gram', metrics['bleu_score_1'] * 100],
            ['BLEU Score 2-gram', metrics['bleu_score_2'] * 100],
            ['BLEU Score 3-gram', metrics['bleu_score_3'] * 100],
            ['Closed Answer Accuracy', metrics['closed_accuracy'] * 100],
        ],
        headers=['Metric', 'Performance (%)'], floatfmt='.2f'
    )
    accelerator.print(f"\n=== Evaluation Results ({len(predictions)} samples) ===")
    accelerator.print(results_table)


class EarlyStopping:
    """Stops training if validation loss doesn't improve after `patience` epochs."""

    def __init__(self, patience=5, verbose=True, accelerator=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.accelerator = accelerator

    def __call__(self, val_loss, model, optimizer, epoch, args):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch, args)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose and self.accelerator:
                self.accelerator.print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch, args)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, epoch, args):
        if not self.accelerator.is_main_process:
            return
        output_dir = Path(args.save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / "best_checkpoint.pth"
        unwrapped_model = self.accelerator.unwrap_model(model)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": unwrapped_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "val_loss": val_loss,
        }
        torch.save(checkpoint, checkpoint_path)
        self.val_loss_min = val_loss
        if self.verbose and self.accelerator:
            self.accelerator.print(f"Validation loss improved: saving new best model at {checkpoint_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train LLaMA-Adapter for medical VQA")

    # Model / tokenizer
    parser.add_argument("--llm-model-path", type=str, required=True,
                        help="Directory containing the LLaMA tokenizer.model")
    parser.add_argument("--vision-model", type=str, default="biomedclip",
                        choices=["clip", "vit", "resnet18", "resnet50", "medclip", "biomedclip", "biomedclip_simpool"],
                        help="Vision encoder to use")
    parser.add_argument("--llm-layers", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=512)

    # Adapter
    parser.add_argument("--adapter-percentage", type=float, default=0.95)
    parser.add_argument("--adapter-strategy", type=str, default="late", choices=["early", "late"])
    parser.add_argument("--query-len", type=int, default=10)
    parser.add_argument("--adapter-dim", type=int, default=256)

    # Deep prompts
    parser.add_argument("--use-deep-prompts", action="store_true", default=False)
    parser.add_argument("--num-deep-prompt-layers", type=int, default=1)
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--prompt-dim", type=int, default=256)

    # Data
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to the VQA train json (conversations)")
    parser.add_argument("--image-folder", type=str, default="",
                        help="Image folder; leave empty if image paths in the json are absolute/relative-to-cwd")
    parser.add_argument("--val-data-path", type=str, default=None,
                        help="Optional validation json; if omitted, no validation is run")
    parser.add_argument("--phrase-type", type=str, default=None)
    parser.add_argument("--question-type", type=str, default=None)

    # Training
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=0.)
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--phase", type=str, default="finetune", choices=["pretrain", "finetune"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-type", type=str, default="vqa", choices=["vqa", "classification"])

    # Accelerate / DeepSpeed
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--deepspeed-config", type=str, default="./ds_config_zero2.json")
    parser.add_argument("--use-combined-loader", action="store_true", default=False,
                        help="Use the combined multi-dataset dataloader (pretraining)")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to a checkpoint to resume / initialize from")
    return parser.parse_args()


def count_trainable_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total:,}")
    return total


def save_checkpoint(model, optimizer, epoch, args, accelerator):
    """Save model checkpoint with optimizer state (main process only)."""
    if not accelerator.is_main_process:
        return
    output_dir = Path(args.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"checkpoint-epoch-{epoch}.pth"
    unwrapped_model = accelerator.unwrap_model(model)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": unwrapped_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(checkpoint, checkpoint_path)


def build_transform(vision_model):
    if vision_model == "vit":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    if vision_model == "biomedclip":
        _, transform = open_clip.create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        return transform
    if vision_model == "medclip":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5862785803043838], std=[0.27950088968644304]),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=args.deepspeed_config)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)

    if accelerator.is_main_process:
        logger.info(f"Training with config: {args}")
        os.makedirs(args.save_dir, exist_ok=True)

    # Tokenizer (SentencePiece model inside the LLaMA model directory)
    tokenizer_model_path = os.path.join(args.llm_model_path, "tokenizer.model")
    tokenizer = Tokenizer(model_path=tokenizer_model_path)

    transform = build_transform(args.vision_model)

    # ---- Datasets ----
    if args.use_combined_loader:
        logger.info("Using combined dataloader...")
        train_loader = get_combined_dataloader(
            batch_size=args.batch_size,
            shuffle=True,
            transform=transform,
            max_seq_len=args.max_seq_len,
            hf_model_path=args.llm_model_path,
        )
        val_loader = None
        if args.val_data_path:
            val_dataset = VQADataset(
                data_path=args.val_data_path,
                tokenizer=tokenizer,
                image_folder=args.image_folder,
                transform=transform,
                max_seq_len=args.max_seq_len,
                split="test",
                phrase_type=args.phrase_type,
            )
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                    shuffle=False, num_workers=8, pin_memory=True)
    else:
        train_dataset = VQADataset(
            data_path=args.data_path,
            tokenizer=tokenizer,
            image_folder=args.image_folder,
            transform=transform,
            max_seq_len=args.max_seq_len,
            split="train",
            phrase_type=args.phrase_type,
            question_type=args.question_type,
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=8, pin_memory=True, drop_last=False,
                                  persistent_workers=True)
        val_loader = None
        if args.val_data_path:
            val_dataset = VQADataset(
                data_path=args.val_data_path,
                tokenizer=tokenizer,
                image_folder=args.image_folder,
                transform=transform,
                max_seq_len=args.max_seq_len,
                split="test",
                phrase_type=args.phrase_type,
                question_type=args.question_type,
            )
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=8, pin_memory=True, drop_last=False,
                                    persistent_workers=True)

    # ---- Model ----
    args.batch_size = train_loader.batch_size
    model = LLamaAdapter(args)
    count_trainable_parameters(model)

    # ---- Optimizer (parameter groups) ----
    param_groups = [
        {"params": [], "lr": args.learning_rate, "weight_decay": args.weight_decay, "name": "embedding"},
        {"params": [], "lr": args.learning_rate, "weight_decay": args.weight_decay, "name": "prefix_prompts"},
        {"params": [], "lr": args.learning_rate, "weight_decay": args.weight_decay, "name": "llama_no_decay"},
        {"params": [], "lr": args.learning_rate, "weight_decay": args.weight_decay, "name": "llama_decay"},
    ]
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(x in name for x in ["feature_proj", "feature_norm", "adapter_query", "prompt_projections",
                                   "clip_proj", "clip_proj_norm", "visual_query", "visual_blocks",
                                   "visual_proj", "visual_proj_norm"]):
            param_groups[0]["params"].append(param)
        elif "base_prompts" in name:
            param_groups[1]["params"].append(param)
        elif len(param.shape) == 1 or name.endswith(".bias"):
            param_groups[2]["params"].append(param)
        else:
            param_groups[3]["params"].append(param)

    optimizer = torch.optim.AdamW(
        [pg for pg in param_groups if len(pg["params"]) > 0],
        betas=(0.9, 0.95),
    )

    # ---- Resume / init from checkpoint ----
    start_epoch = 0
    if args.resume_from:
        if accelerator.is_main_process:
            logger.info(f"Loading checkpoint from {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        if accelerator.is_main_process:
            logger.info(f"Resuming from epoch {start_epoch}")

    if val_loader is not None:
        model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    else:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    model.train()
    early_stopping = EarlyStopping(patience=100, verbose=True, accelerator=accelerator) if val_loader else None

    avg_val_loss = None
    for epoch in tqdm(range(start_epoch, args.epochs)):
        model.train()
        train_loss = 0
        for step, batch in enumerate(train_loader):
            lr_sched.adjust_learning_rate(optimizer, step / len(train_loader) + epoch, args)
            outputs = model(
                pixel_values=batch["pixel_values"].to(accelerator.device),
                input_ids=batch["input_ids"].to(accelerator.device),
                attention_mask=batch["attention_mask"].to(accelerator.device),
                labels=batch["labels"].to(accelerator.device),
            )
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            accelerator.backward(loss)
            train_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = train_loss / len(train_loader)
        accelerator.print(f"Epoch {epoch+1}/{args.epochs} completed. Average loss: {avg_train_loss:.4f}")

        if val_loader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_outputs = model(
                        pixel_values=val_batch["pixel_values"],
                        input_ids=val_batch["input_ids"],
                        attention_mask=val_batch["attention_mask"],
                        labels=val_batch["labels"],
                    )
                    val_batch_loss = val_outputs["loss"] if isinstance(val_outputs, dict) else val_outputs
                    val_loss += val_batch_loss.item()
            avg_val_loss = val_loss / len(val_loader)
            accelerator.print(f"Epoch {epoch+1}/{args.epochs}: Validation Loss: {avg_val_loss:.4f}")

            if (epoch + 1) % 3 == 0:
                accelerator.print(f"\n{'='*50}")
                accelerator.print(f"COMPREHENSIVE EVALUATION - EPOCH {epoch+1}")
                accelerator.print(f"{'='*50}")
                eval_metrics = evaluate_model_metrics_complete(
                    args, model=model, val_loader=val_loader,
                    tokenizer=tokenizer, accelerator=accelerator,
                )
                if accelerator.is_main_process and eval_metrics:
                    accelerator.print("Key Metrics Summary:")
                    accelerator.print(f"  - Exact Match (Open): {eval_metrics.get('exact_match_open', 0)*100:.2f}%")
                    accelerator.print(f"  - F1 Score: {eval_metrics.get('f1_score', 0)*100:.2f}%")
                    accelerator.print(f"  - BLEU Score: {eval_metrics.get('bleu_score', 0)*100:.2f}%")
                    accelerator.print(f"  - Closed Accuracy: {eval_metrics.get('closed_accuracy', 0)*100:.2f}%")
                    accelerator.print(f"  - Total Samples Evaluated: {eval_metrics.get('num_total', 0)}")
                accelerator.wait_for_everyone()

            early_stopping(avg_val_loss, model, optimizer, epoch, args)
            if early_stopping.early_stop:
                accelerator.print("Early stopping triggered. Ending training.")
                break

        save_checkpoint(model, optimizer, epoch + 1, args, accelerator)

    # ---- Final save ----
    if accelerator.is_main_process:
        final_output_dir = os.path.join(args.save_dir, "final")
        os.makedirs(final_output_dir, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)

        torch.save({
            "epoch": args.epochs,
            "model_state_dict": unwrapped_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "val_loss": avg_val_loss,
        }, os.path.join(final_output_dir, "final_checkpoint.pth"))

        accelerator.save(unwrapped_model.state_dict(), os.path.join(final_output_dir, "pytorch_model.bin"))

        with open(os.path.join(final_output_dir, "config.json"), 'w') as f:
            json.dump({
                "model_type": args.vision_model,
                "llm_layers": args.llm_layers,
                "adapter_percentage": args.adapter_percentage,
                "adapter_strategy": args.adapter_strategy,
                "query_len": args.query_len,
                "tokenizer_model_path": tokenizer_model_path,
            }, f, indent=2)
        logger.info(f"Training completed. Final model saved to {final_output_dir}")


if __name__ == "__main__":
    main()
