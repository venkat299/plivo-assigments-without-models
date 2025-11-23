"""
Dynamic quantization helper for HF token classification models.

Usage:
  python src/quantize_model.py --model_dir out --out_dir out_quant
"""
import argparse
import os

import torch
torch.backends.quantized.engine = "qnnpack"
from transformers import AutoModelForTokenClassification, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Path to fp32 model directory")
    ap.add_argument("--out_dir", required=True, help="Path to save quantized model")
    args = ap.parse_args()

    print(f"Loading model from {args.model_dir}")
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.eval()

    print("Applying dynamic quantization (Linear layers)...")
    q_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Saving quantized model to {args.out_dir}")
    # Direct save_pretrained may fail on quantized modules; save state_dict + config manually.
    torch.save(q_model.state_dict(), os.path.join(args.out_dir, "pytorch_model.bin"))
    q_model.config.save_pretrained(args.out_dir)

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    tok.save_pretrained(args.out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
