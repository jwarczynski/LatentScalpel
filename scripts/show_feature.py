"""Quick script to display top-activating examples for a given feature.

Highlights the specific token at `token_position` using ANSI colors.

Usage:
    uv run python scripts/show_feature.py <path> <feature_idx> [num_show]
"""
import json
import sys

from datasets import load_dataset
from transformers import AutoTokenizer


# ANSI escape codes
RED_BG = "\033[41m\033[97m"  # red background, white text
RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"


def highlight_token_in_context(
    text: str,
    token_position: int,
    tokenizer,
    context_tokens: int = 15,
) -> str:
    """Tokenize text and return a string with the target token highlighted.

    Shows `context_tokens` tokens before and after the target token.
    """
    encoding = tokenizer(
        text, padding=True, truncation=True, max_length=512, return_tensors=None,
    )
    input_ids = encoding["input_ids"]

    if token_position < 0 or token_position >= len(input_ids):
        return f"  [token_position {token_position} out of range for {len(input_ids)} tokens]"

    # Decode individual tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    target_token = tokens[token_position]

    # Build context window
    start = max(1, token_position - context_tokens)  # skip [CLS]
    end = min(len(tokens), token_position + context_tokens + 1)

    # Skip padding tokens at the end
    while end > start and tokens[end - 1] in ("[PAD]", tokenizer.pad_token):
        end -= 1

    parts = []
    if start > 1:
        parts.append("...")
    for i in range(start, end):
        tok = tokens[i]
        if tok in ("[PAD]", "[CLS]", "[SEP]"):
            continue
        # Clean up wordpiece prefix
        display = tok.replace("##", "")
        if i == token_position:
            parts.append(f"{RED_BG} {display} {RESET}")
        else:
            parts.append(display)
    if end < len(tokens) and tokens[end] not in ("[PAD]", "[SEP]"):
        parts.append("...")

    highlighted_line = " ".join(parts)
    return f"  Token: {BOLD}{target_token}{RESET} (pos {token_position}/{len(input_ids)})\n  {highlighted_line}"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "experiments/results/top_examples/layer_01_top_examples.json"
    feat_idx = sys.argv[2] if len(sys.argv) > 2 else "0"
    num_show = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    with open(path) as f:
        data = json.load(f)

    meta = data["metadata"]
    print(f"Dataset: {meta['dataset_name']} / {meta['dataset_split']}")
    print(f"Layer: {meta['layer_idx']}, top_k: {meta['top_k']}")
    print()

    feat = data["features"].get(feat_idx)
    if feat is None:
        print(f"Feature {feat_idx} not found. Available: {list(data['features'].keys())[:20]}...")
        return

    print(f"Feature {feat_idx}: {len(feat)} entries")
    print(f"Loading dataset {meta['dataset_name']}...")
    ds = load_dataset(meta["dataset_name"], split=meta["dataset_split"])

    print("Loading tokenizer bert-base-uncased...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    for i, entry in enumerate(feat[:num_show]):
        eid = entry["example_id"]
        act = entry["activation"]
        ts = entry["timestep"]
        tp_ = entry["token_position"]
        text = ds[eid]["document"]

        print(f"\n{'='*80}")
        print(f"#{i+1}  example_id={eid}  activation={act:.4f}  timestep={ts}  token_pos={tp_}")
        print(f"{'='*80}")

        # Show highlighted token in context
        print(highlight_token_in_context(text, tp_, tokenizer))

        # Show truncated raw text
        print(f"\n{DIM}--- full text ---{RESET}")
        if len(text) > 500:
            text = text[:500] + "..."
        print(f"{DIM}{text}{RESET}")


if __name__ == "__main__":
    main()
