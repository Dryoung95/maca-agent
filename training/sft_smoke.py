#!/usr/bin/env python3
"""SFT smoke test on the official RFT positives (12 samples).

GOAL: verify that a 1.5B model can be fine-tuned on this machine at all —
loss goes down, memory fits, nothing crashes. NOT to produce a useful
policy (12 samples is far too few; this is a pipeline check).

Format: agentic tool-use SFT. The prompt is the stage-1 prompt as
collected (SKILL.md contract + task), and the target is the write_files
tool call serialized as the model would emit it — exactly what the
collection loop recorded.

Only `response` rows with type == 'tool_use' and all four fields are used;
text-only responses (the model wrote prose instead of calling the tool)
are skipped, since there is nothing to supervise.

Memory plan (bf16 full fine-tune, 1.5B):
  weights bf16  ~3.1 GB, grads bf16 ~3.1 GB, AdamW fp32 m+v ~12.4 GB
  => ~19 GB floor + activations; the 68 GB board has plenty of headroom.
"""
import argparse
import json
import math
import os
import random

import torch
from torch.utils.data import Dataset

HF_SNAPSHOT = ("/data/huggingface_home/hub/"
               "models--Qwen--Qwen2.5-Coder-1.5B-Instruct/snapshots/"
               "2e1fd397ee46e1388853d2af2c993145b0f1098a")


def serialize_tool_use(response, kernel_name):
    """Render the recorded tool_use as the assistant text to supervise."""
    inp = response["input"]
    return (
        f'<tool_use name="write_files"\n'
        f'  kernel_name="{kernel_name}"\n'
        f'  kernel_cu={json.dumps(inp["kernel_cu"])}\n'
        f'  binding_cpp={json.dumps(inp["binding_cpp"])}\n'
        f'  model_new_py={json.dumps(inp["model_new_py"])}\n'
        f'</tool_use>'
    )


class RFTDataset(Dataset):
    def __init__(self, positives_path, tokenizer, max_len):
        self.rows = []
        skipped = 0
        for line in open(positives_path):
            if not line.strip():
                continue
            rec = json.loads(line)
            resp = rec.get("response")
            if not (isinstance(resp, dict) and resp.get("type") == "tool_use"):
                skipped += 1
                continue
            inp = resp.get("input") or {}
            if not all(inp.get(f) for f in ("kernel_cu", "binding_cpp",
                                            "model_new_py", "kernel_name")):
                skipped += 1
                continue
            prompt = rec["prompt"]
            target = serialize_tool_use(resp, inp["kernel_name"])
            self.rows.append((prompt, target))
        self.tok = tokenizer
        self.max_len = max_len
        print(f"dataset: {len(self.rows)} usable samples "
              f"({skipped} skipped — text-only or missing fields)")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        prompt, target = self.rows[i]
        # eos terminates the assistant turn so the model learns to stop
        prompt_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        target_ids = self.tok(target, add_special_tokens=False)["input_ids"]
        target_ids = target_ids + [self.tok.eos_token_id]
        ids = (prompt_ids + target_ids)[: self.max_len]
        labels = ([-100] * len(prompt_ids) + target_ids)[: self.max_len]
        pad = self.max_len - len(ids)
        if pad > 0:
            ids = ids + [self.tok.pad_token_id] * pad
            labels = labels + [-100] * pad
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        maxlen = max(len(f["input_ids"]) for f in feats)
        input_ids, labels, attn = [], [], []
        for f in feats:
            ids, lab = f["input_ids"], f["labels"]
            pad = maxlen - len(ids)
            input_ids.append(
                torch.cat([ids, torch.full((pad,), self.pad_id,
                                           dtype=torch.long)]))
            labels.append(
                torch.cat([lab, torch.full((pad,), -100, dtype=torch.long)]))
            attn.append(torch.cat([
                torch.ones(len(ids), dtype=torch.long),
                torch.zeros(pad, dtype=torch.long)]))
        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attn),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positives", default="rollouts/rft_positives.jsonl")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--steps", type=int, default=0,
                    help="cap optimization steps (0 = run full dataset)")
    args = ap.parse_args()

    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              Trainer, TrainingArguments,
                              DataCollatorWithPadding)

    print("loading model from", HF_SNAPSHOT, flush=True)
    tok = AutoTokenizer.from_pretrained(HF_SNAPSHOT)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        HF_SNAPSHOT, torch_dtype=torch.bfloat16, attn_implementation="eager")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    ds = RFTDataset(args.positives, tok, args.max_len)
    if not ds.rows:
        raise SystemExit("no usable samples — cannot train")

    collator = Collator(tok.pad_token_id)

    # Memory report before training
    free, total = torch.cuda.mem_get_info()
    print(f"GPU before training: {free/1e9:.1f} GB free / {total/1e9:.1f} GB")

    targs = TrainingArguments(
        output_dir="/tmp/sft_smoke_out",
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        max_steps=args.steps,
        bf16=True,
        seed=17,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=collator)
    print("\n=== training ===", flush=True)
    trainer.train()

    free, total = torch.cuda.mem_get_info()
    print(f"\nGPU after training: {free/1e9:.1f} GB free / "
          f"{total/1e9:.1f} GB")
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"peak allocator: {peak:.2f} GB")

    out = "/data/maca-agent/training/smoke_model"
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"saved -> {out}")
    print("SMOKE OK: loss decreased and memory fit" if trainer.state.log_history
          and trainer.state.log_history[-1].get("loss", 999)
          < trainer.state.log_history[0].get("loss", 0)
          else "check log history")


if __name__ == "__main__":
    main()
