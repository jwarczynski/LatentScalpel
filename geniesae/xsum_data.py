"""XSum dataset loading and tokenization for PLAID fine-tuning.

Reads line-aligned .src/.tgt files, tokenizes with PLAID's 32K BPE tokenizer,
and constructs [article | SEP | summary | PAD] sequences.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import tokenizers
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl

logger = logging.getLogger(__name__)

# Mapping from split name to file prefix
_SPLIT_PREFIXES = {
    "train": "train",
    "dev": "dev",
    "test": "test",
}


class XSumDataset(Dataset):
    """Single-split XSum dataset with PLAID tokenization.

    Supports two format modes:
    - "sep" (default): [article | SEP | summary | PAD]
    - "template": [article \\n\\nTL;DR: summary | PAD]
      Uses natural text template instead of a special SEP token.
      boundary_idx points to the first summary token after the template.
    """

    def __init__(
        self,
        src_path: Path,
        tgt_path: Path | None,
        tokenizer: tokenizers.Tokenizer,
        seq_len: int,
        max_summary_len: int,
        sep_token_id: int,
        pad_token_id: int,
        format_mode: str = "sep",
    ) -> None:
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_summary_len = max_summary_len
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        self.format_mode = format_mode

        # Pre-tokenize the prompt suffix for template mode
        if format_mode == "template":
            self._prefix_ids = tokenizer.encode("ARTICLE:").ids
            self._suffix_ids = tokenizer.encode("\n\nSUMMARY:").ids
        else:
            self._prefix_ids = []
            self._suffix_ids = []

        # Read source lines
        self.src_lines = src_path.read_text().strip().split("\n")
        if not self.src_lines or (len(self.src_lines) == 1 and not self.src_lines[0]):
            raise ValueError(f"Empty source file for split: {src_path}")

        # Read target lines (optional)
        self.tgt_lines: list[str] | None = None
        if tgt_path is not None and tgt_path.exists():
            self.tgt_lines = tgt_path.read_text().strip().split("\n")

    def __len__(self) -> int:
        return len(self.src_lines)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        article_text = self.src_lines[idx]
        article_ids = self.tokenizer.encode(article_text).ids

        if self.tgt_lines is not None:
            summary_text = self.tgt_lines[idx]
            summary_ids = self.tokenizer.encode(summary_text).ids

            # Truncate summary if needed
            if len(summary_ids) > self.max_summary_len:
                summary_ids = summary_ids[: self.max_summary_len]

            if self.format_mode == "template":
                # Format: ARTICLE: <article> \n\nSUMMARY: <summary>
                # boundary_idx = first token of <summary>
                prefix = self._prefix_ids  # "ARTICLE:"
                suffix = self._suffix_ids  # "\n\nSUMMARY:"
                overhead = len(prefix) + len(suffix) + len(summary_ids)
                article_budget = self.seq_len - overhead
                if article_budget < 0:
                    article_budget = 0
                if len(article_ids) > article_budget:
                    article_ids = article_ids[:article_budget]

                sequence = prefix + article_ids + suffix + summary_ids
                # boundary_idx = position of first summary token
                boundary_idx = len(prefix) + len(article_ids) + len(suffix)
            else:
                # Format: [article | SEP | summary]
                article_budget = self.seq_len - len(summary_ids) - 1
                if article_budget < 0:
                    article_budget = 0
                if len(article_ids) > article_budget:
                    article_ids = article_ids[:article_budget]

                sequence = article_ids + [self.sep_token_id] + summary_ids
                boundary_idx = len(article_ids)  # position of SEP
        else:
            if len(article_ids) > self.seq_len:
                article_ids = article_ids[: self.seq_len]
            sequence = article_ids
            boundary_idx = len(article_ids)

        # Pad to seq_len
        real_len = len(sequence)
        if real_len < self.seq_len:
            sequence = sequence + [self.pad_token_id] * (self.seq_len - real_len)
        else:
            sequence = sequence[: self.seq_len]
            real_len = self.seq_len

        # Attention mask: 1 for real tokens, 0 for padding
        attention_mask = [1] * real_len + [0] * (self.seq_len - real_len)

        return {
            "token_ids": torch.tensor(sequence, dtype=torch.long),
            "boundary_idx": torch.tensor(boundary_idx, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class XSumDataModule(pl.LightningDataModule):
    """Lightning DataModule for XSum with PLAID tokenization."""

    def __init__(
        self,
        data_dir: str,
        seq_len: int = 512,
        max_summary_len: int = 64,
        batch_size: int = 8,
        num_workers: int = 4,
        tokenizer_path: str | None = None,
        format_mode: str = "sep",
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.max_summary_len = max_summary_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer_path = tokenizer_path
        self.format_mode = format_mode

        self.tokenizer: tokenizers.Tokenizer | None = None
        self.sep_token_id: int = 0
        self.pad_token_id: int = 0

        self.train_dataset: XSumDataset | None = None
        self.val_dataset: XSumDataset | None = None
        self.test_dataset: XSumDataset | None = None

    def _load_tokenizer(self) -> tokenizers.Tokenizer:
        """Load the PLAID BPE tokenizer."""
        if self.tokenizer_path:
            return tokenizers.Tokenizer.from_file(self.tokenizer_path)
        # Try common locations
        for candidate in [
            Path("models/plaid/plaid1b_weights/tokenizer.json"),
            Path("tokenizer.json"),
        ]:
            if candidate.exists():
                return tokenizers.Tokenizer.from_file(str(candidate))
        raise FileNotFoundError(
            "PLAID tokenizer not found. Provide tokenizer_path in config."
        )

    def _resolve_sep_pad(self, tok: tokenizers.Tokenizer) -> tuple[int, int]:
        """Resolve SEP and PAD token IDs from the tokenizer vocabulary.

        The PLAID 32K BPE tokenizer has a single special token:
        <|endoftext_R9VQqF0Ag7|> at index 0, which serves as both
        end-of-text and document separator. We reuse it as SEP.
        For PAD we also use index 0 (same as EOT) since there's no
        dedicated pad token — padding positions are masked out via
        attention_mask anyway.
        """
        vocab = tok.get_vocab()
        # Use the EOT/special token as SEP (index 0 in PLAID tokenizer)
        for sep_candidate in ["<|endoftext_R9VQqF0Ag7|>", "<sep>", "[SEP]", "<|sep|>"]:
            if sep_candidate in vocab:
                sep_id = vocab[sep_candidate]
                break
        else:
            sep_id = 0  # default to index 0

        # PAD: use same as EOT (masked out by attention_mask)
        pad_id = sep_id

        return sep_id, pad_id

    def setup(self, stage: str | None = None) -> None:
        """Load tokenizer and create datasets for train/val/test splits."""
        if self.tokenizer is None:
            self.tokenizer = self._load_tokenizer()
            self.sep_token_id, self.pad_token_id = self._resolve_sep_pad(self.tokenizer)

        for split_name, attr_name in [
            ("train", "train_dataset"),
            ("dev", "val_dataset"),
            ("test", "test_dataset"),
        ]:
            if getattr(self, attr_name) is not None:
                continue

            src_path = self.data_dir / f"{split_name}.src"
            tgt_path = self.data_dir / f"{split_name}.tgt"

            if not src_path.exists():
                if split_name == "train":
                    raise FileNotFoundError(f"Missing source file: {src_path}")
                logger.warning("Skipping split %s: %s not found", split_name, src_path)
                continue

            dataset = XSumDataset(
                src_path=src_path,
                tgt_path=tgt_path if tgt_path.exists() else None,
                tokenizer=self.tokenizer,
                seq_len=self.seq_len,
                max_summary_len=self.max_summary_len,
                sep_token_id=self.sep_token_id,
                pad_token_id=self.pad_token_id,
                format_mode=self.format_mode,
            )
            setattr(self, attr_name, dataset)

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None, "Call setup() first"
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None, "Call setup() first"
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None, "Call setup() first"
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
