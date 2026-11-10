#!/usr/bin/env python3
"""Run a local CPU inference using a downloaded Hugging Face causal language model."""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "SmolLM2-135M-Instruct"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--prompt",
        default="What is 2 + 2? Answer briefly.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be greater than zero")
    if not args.model.is_dir():
        parser.error(f"model directory does not exist: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.float32,
    )
    model.to("cpu")
    model.eval()

    messages = [{"role": "user", "content": args.prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(formatted_prompt, return_tensors="pt")

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output[0, inputs["input_ids"].shape[1] :]
    print(tokenizer.decode(generated_tokens, skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
