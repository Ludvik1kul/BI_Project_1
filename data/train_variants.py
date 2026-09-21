"""Resumable Lightning fine-tuning for the Llama model variants."""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "last_layer"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data", default="./data/train_data.jsonl")
    parser.add_argument("--max-time", default=None, help="Lightning duration, e.g. 00:30:00")
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--resume", default="auto", help="Checkpoint path, 'auto', or 'none'")
    return parser.parse_args()


def to_multitask(example):
    if "prompt" in example and "target" in example:
        return [example]
    input_prompt = example["input_prompt"]
    return [
        {"prompt": "predict: " + input_prompt, "target": example["answer"]},
        {"prompt": "explain: " + input_prompt, "target": example["answer_rationale"]},
    ]


class TextDataset(Dataset):
    def __init__(self, records, tokenizer, max_length=512):
        self.examples = []
        for record in records:
            prompt_tokens = tokenizer(record["prompt"] + " ", add_special_tokens=False)["input_ids"]
            target_tokens = tokenizer(
                record["target"] + tokenizer.eos_token,
                add_special_tokens=False,
            )["input_ids"]
            input_ids = (prompt_tokens + target_tokens)[:max_length]
            target_start = min(len(prompt_tokens), len(input_ids))
            attention_mask = [1] * len(input_ids)
            labels = [-100] * target_start + input_ids[target_start:]
            padding_length = max_length - len(input_ids)
            input_ids += [tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length
            labels += [-100] * padding_length
            encoded = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
            self.examples.append(encoded)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return {key: torch.tensor(value, dtype=torch.long) for key, value in self.examples[index].items()}


class ResumableRandomSampler(Sampler):
    """Shuffled sampler whose permutation and cursor are checkpointable."""

    def __init__(self, data_source, seed=42):
        self.data_source = data_source
        self.seed = seed
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.epoch = 0
        self.position = 0
        self.permutation = []
        self._start_epoch()

    def _start_epoch(self):
        self.permutation = torch.randperm(len(self.data_source), generator=self.generator).tolist()
        self.position = 0

    def __iter__(self):
        while self.position < len(self.permutation):
            index = self.permutation[self.position]
            self.position += 1
            yield index
        self.epoch += 1
        self._start_epoch()

    def __len__(self):
        return len(self.data_source)

    def state_dict(self):
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "position": self.position,
            "permutation": self.permutation,
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state_dict):
        self.seed = state_dict["seed"]
        self.epoch = state_dict["epoch"]
        self.position = state_dict["position"]
        self.permutation = state_dict["permutation"]
        self.generator.set_state(state_dict["generator_state"])


class ResumableDataLoader(DataLoader):
    """DataLoader state wrapper recognized by Lightning's checkpoint logic."""

    def state_dict(self):
        return {"sampler": self.sampler.state_dict()}

    def load_state_dict(self, state_dict):
        self.sampler.load_state_dict(state_dict["sampler"])


def create_lightning_model_class():
    import lightning.pytorch as pl

    class CausalLanguageModel(pl.LightningModule):
        def __init__(self, model, learning_rate=1e-6):
            super().__init__()
            self.model = model
            self.learning_rate = learning_rate

        def forward(self, **batch):
            return self.model(**batch)

        def training_step(self, batch, batch_index):
            loss = self(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at batch {batch_index}: {loss.item()}")
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            return loss

        def configure_optimizers(self):
            trainable_parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
            return torch.optim.AdamW(trainable_parameters, lr=self.learning_rate)

    return pl, CausalLanguageModel


def create_loss_logger_class():
    import lightning.pytorch as pl

    class LossHistoryLogger(pl.Callback):
        def __init__(self, path):
            self.path = Path(path)
            self.started_at = time.time()
            self.last_logged_step = -1

        def on_fit_start(self, trainer, pl_module):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                with self.path.open(newline="", encoding="utf-8") as source:
                    rows = list(csv.DictReader(source))
                if rows:
                    self.last_logged_step = max(int(row["global_step"]) for row in rows)
            else:
                with self.path.open("w", newline="", encoding="utf-8") as target:
                    csv.writer(target).writerow(["global_step", "epoch", "loss", "elapsed_seconds"])

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_index):
            step = trainer.global_step
            if step == 0:
                return
            if step <= self.last_logged_step:
                return
            loss = outputs.detach().float().item() if torch.is_tensor(outputs) else None
            if loss is None:
                return
            with self.path.open("a", newline="", encoding="utf-8") as target:
                csv.writer(target).writerow([
                    step,
                    trainer.current_epoch,
                    f"{loss:.8f}",
                    f"{time.time() - self.started_at:.2f}",
                ])
                target.flush()
            self.last_logged_step = step

    return LossHistoryLogger


def find_resume_checkpoint(output_dir, resume):
    if resume == "none":
        return None
    if resume != "auto":
        return resume
    last_checkpoint = Path(output_dir) / "checkpoints" / "last.ckpt"
    return str(last_checkpoint) if last_checkpoint.exists() else None


def main():
    args = parse_args()
    output_dir = Path(args.output)
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)

    if args.mode == "last_layer":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.lm_head.parameters():
            parameter.requires_grad = True

    raw_records = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines()]
    train_records = [task for record in raw_records for task in to_multitask(record)]
    train_dataset = TextDataset(train_records, tokenizer)
    sampler = ResumableRandomSampler(train_dataset)
    train_loader = ResumableDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
    )

    pl, CausalLanguageModel = create_lightning_model_class()
    lightning_model = CausalLanguageModel(model)
    LossHistoryLogger = create_loss_logger_class()
    loss_logger = LossHistoryLogger(output_dir / "loss_history.csv")
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="last",
        auto_insert_metric_name=False,
        every_n_train_steps=1,
        save_last=False,
        save_top_k=1,
        save_on_train_epoch_end=False,
        enable_version_counter=False,
    )
    logger = pl.loggers.CSVLogger(save_dir=output_dir / "logs", name=args.mode)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    precision = "bf16-mixed" if accelerator == "gpu" else "32-true"
    trainer = pl.Trainer(
        default_root_dir=output_dir,
        accelerator=accelerator,
        devices=1,
        precision=precision,
        max_epochs=args.max_epochs,
        max_time=args.max_time,
        accumulate_grad_batches=args.gradient_accumulation,
        callbacks=[checkpoint, loss_logger],
        logger=logger,
        log_every_n_steps=1,
        enable_progress_bar=True,
    )

    resume_checkpoint = find_resume_checkpoint(output_dir, args.resume)
    print(f"Training {args.mode} variant with {len(train_dataset)} task records.")
    print(f"Checkpointing every optimizer update under {checkpoint_dir}.")
    print("Using a resumable sampler; dataloader position will be restored from the checkpoint.")
    print(f"Resuming from: {resume_checkpoint or 'model weights'}")
    trainer.fit(lightning_model, train_loader, ckpt_path=resume_checkpoint)

    model.save_pretrained(output_dir / "final")
    tokenizer.save_pretrained(output_dir / "final")
    print(f"Training stopped or completed. Latest weights saved to {output_dir / 'final'}.")


if __name__ == "__main__":
    main()
