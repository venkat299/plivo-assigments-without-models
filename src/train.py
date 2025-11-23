import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from dataset import PIIDataset, collate_batch
from labels import LABELS
from model import create_model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--dev", default="data/dev.jsonl")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--class_weights", action="store_true", help="Use inverse-frequency class weights in loss.")
    ap.add_argument(
        "--loss_type",
        choices=["ce", "focal"],
        default="ce",
        help="Cross-entropy (ce) or focal loss (focal) for precision-oriented training.",
    )
    ap.add_argument("--focal_gamma", type=float, default=2.0, help="Gamma for focal loss (ignored for CE).")
    ap.add_argument("--clip_grad_norm", type=float, default=0.0, help="Clip grad norm if > 0.")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = PIIDataset(args.train, tokenizer, LABELS, max_length=args.max_length, is_train=True)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id),
    )

    model = create_model(args.model_name)
    model.to(args.device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    # Optional class weights to down-weight majority classes
    weight_tensor = None
    if args.class_weights:
        label_counts = torch.zeros(len(LABELS))
        for item in train_ds:
            for lid in item["labels"]:
                label_counts[lid] += 1
        # Avoid divide by zero; ignore pad (-100) which is not present here
        label_counts[label_counts == 0] = 1.0
        inv_freq = 1.0 / label_counts
        weights = inv_freq / inv_freq.sum() * len(LABELS)
        weight_tensor = weights.to(args.device)

    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = torch.tensor(batch["input_ids"], device=args.device)
            attention_mask = torch.tensor(batch["attention_mask"], device=args.device)
            labels = torch.tensor(batch["labels"], device=args.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (batch, seq, num_labels)
            loss = compute_loss(
                logits,
                labels,
                loss_type=args.loss_type,
                focal_gamma=args.focal_gamma,
                weight=weight_tensor,
            )

            optimizer.zero_grad()
            loss.backward()
            if args.clip_grad_norm and args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_dl))
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Saved model + tokenizer to {args.out_dir}")


def compute_loss(logits, labels, loss_type="ce", focal_gamma=2.0, weight=None):
    """Supports CE or focal loss; ignores pad tokens (-100)."""
    logits_flat = logits.view(-1, logits.size(-1))
    labels_flat = labels.view(-1)
    if loss_type == "focal":
        log_probs = F.log_softmax(logits_flat, dim=-1)
        ce = F.nll_loss(
            log_probs,
            labels_flat,
            weight=weight,
            reduction="none",
            ignore_index=-100,
        )
        with torch.no_grad():
            pt = torch.exp(-ce)
        loss = ((1 - pt) ** focal_gamma) * ce
        # mask ignored positions to avoid nan
        mask = labels_flat.ne(-100)
        loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        return loss
    else:
        return F.cross_entropy(
            logits_flat,
            labels_flat,
            weight=weight,
            ignore_index=-100,
        )


if __name__ == "__main__":
    main()
