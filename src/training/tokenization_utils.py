from __future__ import annotations
from pathlib import Path
import re
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

def make_tokenized_datasets(model_name: str, max_length: int, ds_train_raw, ds_val_raw, ds_dev_raw):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def tok_fn(batch):
        texts = [str(x) if x is not None else "" for x in batch["text"]]
        tokenized = tok(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        if "label_bin" in batch:
            tokenized["labels"] = [float(x) for x in batch["label_bin"]]
        return dict(tokenized)

    # Tokenize datasets & truncate/padd them to max length
    ds_train = ds_train_raw.map(tok_fn, batched=True)
    ds_val   = ds_val_raw.map(tok_fn, batched=True)
    ds_dev   = ds_dev_raw.map(tok_fn, batched=True)

    if "labels" not in ds_train.column_names and "label_bin" in ds_train.column_names:
        ds_train = ds_train.rename_column("label_bin", "labels")
    if "labels" not in ds_val.column_names and "label_bin" in ds_val.column_names:
        ds_val = ds_val.rename_column("label_bin", "labels")
    if "labels" not in ds_dev.column_names and "label_bin" in ds_dev.column_names:
        ds_dev = ds_dev.rename_column("label_bin", "labels")

    return tok, ds_train, ds_val, ds_dev


def clean_and_prune_by_tokens(df, tokenizer_name, text_col="text", label_col="label_bin", max_pos_tokens=None):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    df[text_col] = df[text_col].astype(str).map(clean_text)
    if max_pos_tokens is not None:
        # Compute token counts
        token_counts = df[text_col].map(lambda x: len(tokenizer.encode(x, add_special_tokens=True)))
        mask = ~((df[label_col] == 1) & (token_counts > max_pos_tokens))
        df = df[mask].reset_index(drop=True)
    return df

def clean_text(t):
    t = re.sub(r"&\w+;", " ", t)
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


# :::::::::: Token cache building and dataset class :::::::::



def _coerce_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    if isinstance(x, str):
        return x
    return str(x)


@dataclass(frozen=True)
class TokenCacheMeta:
    model_key: str
    model_name: str
    max_len: int
    rows: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_key": self.model_key,
            "model_name": self.model_name,
            "max_len": int(self.max_len),
            "rows": int(self.rows),
        }


def build_token_cache(
    df: pd.DataFrame,
    *,
    tokenizer,
    max_len: int,
    cache_path: Path | str,
    model_key: str = "",
    model_name: str = "",
    text_col: str = "text",
    label_col: str = "label_bin",
    spans_col: str = "span_ranges",
) -> Path:
    """
    Pre-tokenize once and save tensors to disk.

    Saves:
      - input_ids:       (N, T) long
      - attention_mask:  (N, T) long/bool
      - token_type_ids:  (N, T) long or None
      - token_labels:    (N, T) float (0/1)
      - token_loss_mask: (N, T) bool (real tokens only; excludes padding+specials)
      - paragraph_label: (N,)   float (0/1)
      - meta: dict

    Notes:
      - Uses tokenizer(..., return_offsets_mapping=True) so requires a *fast* tokenizer.
      - Token labeling is overlap-based: token is positive if its char span overlaps any annotated span.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    df = df.reset_index(drop=True)

    input_ids_list = []
    attention_mask_list = []
    token_type_ids_list = []
    token_labels_list = []
    token_loss_mask_list = []
    paragraph_label_list = []

    any_token_type_ids = False

    for i in range(len(df)):
        row = df.iloc[i]
        text = _coerce_text(row.get(text_col, ""))

        enc = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=int(max_len),
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        offsets = enc["offset_mapping"][0]  # (T,2)
        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]
        token_type_ids = enc["token_type_ids"][0] if "token_type_ids" in enc else None
        if token_type_ids is not None:
            any_token_type_ids = True

        token_labels = torch.zeros(int(max_len), dtype=torch.float32)

        # real tokens: not padding AND not special tokens (specials often have offset (0,0))
        is_real_token = (attention_mask == 1) & (offsets[:, 1] > offsets[:, 0])

        spans = row.get(spans_col, [])
        if not isinstance(spans, list):
            spans = []

        # overlap assignment
        off = offsets.tolist()
        for j, (start, end) in enumerate(off):
            if not bool(is_real_token[j].item()):
                continue
            for s, e in spans:
                # token [start,end) overlaps span [s,e)
                if start < e and end > s:
                    token_labels[j] = 1.0
                    break

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        token_labels_list.append(token_labels)
        token_loss_mask_list.append(is_real_token.to(torch.bool))
        paragraph_label_list.append(torch.tensor(float(row.get(label_col, 0.0)), dtype=torch.float32))
        token_type_ids_list.append(token_type_ids)

    token_type_ids_tensor: Optional[torch.Tensor]
    if any_token_type_ids:
        token_type_ids_tensor = torch.stack(
            [
                t if t is not None else torch.zeros(int(max_len), dtype=torch.long)
                for t in token_type_ids_list
            ],
            dim=0,
        )
    else:
        token_type_ids_tensor = None

    meta = TokenCacheMeta(
        model_key=str(model_key),
        model_name=str(model_name),
        max_len=int(max_len),
        rows=int(len(df)),
    ).to_dict()

    cache = {
        "input_ids": torch.stack(input_ids_list, dim=0),
        "attention_mask": torch.stack(attention_mask_list, dim=0),
        "token_type_ids": token_type_ids_tensor,
        "token_labels": torch.stack(token_labels_list, dim=0),
        "token_loss_mask": torch.stack(token_loss_mask_list, dim=0),
        "paragraph_label": torch.stack(paragraph_label_list, dim=0),
        "meta": meta,
    }

    torch.save(cache, cache_path)
    return cache_path


def load_token_cache(cache_path: Path | str, *, map_location: str = "cpu") -> Dict[str, Any]:
    cache_path = Path(cache_path)
    return torch.load(cache_path, map_location=map_location)


class TensorCacheDataset(Dataset):
    """
    Dataset backed by a saved token cache (torch.save dict).

    Returns the same dict keys your model expects:
      input_ids, attention_mask, (optional token_type_ids), token_labels,
      token_loss_mask, paragraph_label
    """

    def __init__(self, cache_path: Path | str):
        self.cache_path = Path(cache_path)
        self.cache = load_token_cache(self.cache_path, map_location="cpu")

        self.input_ids = self.cache["input_ids"]
        self.attention_mask = self.cache["attention_mask"]
        self.token_type_ids = self.cache.get("token_type_ids", None)
        self.token_labels = self.cache["token_labels"]
        self.token_loss_mask = self.cache["token_loss_mask"]
        self.paragraph_label = self.cache["paragraph_label"]

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        batch = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "token_labels": self.token_labels[idx],
            "token_loss_mask": self.token_loss_mask[idx],
            "paragraph_label": self.paragraph_label[idx],
        }
        if self.token_type_ids is not None:
            batch["token_type_ids"] = self.token_type_ids[idx]
        return batch


def ensure_token_cache(
    df: pd.DataFrame,
    *,
    tokenizer,
    max_len: int,
    cache_path: Path | str,
    model_key: str = "",
    model_name: str = "",
    rebuild: bool = False,
    **kwargs,
) -> Path:
    """
    Convenience wrapper: build cache if missing (or rebuild=True).
    """
    cache_path = Path(cache_path)
    if rebuild or (not cache_path.exists()):
        return build_token_cache(
            df,
            tokenizer=tokenizer,
            max_len=max_len,
            cache_path=cache_path,
            model_key=model_key,
            model_name=model_name,
            **kwargs,
        )
    return cache_path




# Old version prior to cached tokenisation

class PCLTokenDataset(Dataset):
    """
    Expects df columns:
      - text (str)
      - label_bin (0/1)
      - span_ranges (list[(start,end)])  (can be empty list)
    Produces:
      - input_ids, attention_mask
      - token_type_ids (only if tokenizer returns it; needed for ALBERT/BERT-like)
      - token_labels (0/1 per token)
      - token_loss_mask (bool mask for real tokens only; excludes padding + specials)
      - paragraph_label (float)
    """
    def __init__(self, df, max_len, tokenizer):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text = row["text"]

        # ---- guard: tokenizer requires str / list[str] ----
        # If text is NaN/None/float, turn into empty string (or str(text) if you prefer).
        if text is None:
            text = ""
        elif isinstance(text, float):
            # catches NaN too (np.nan is float)
            if np.isnan(text):
                text = ""
            else:
                text = str(text)
        elif not isinstance(text, str):
            text = str(text)

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        offsets = enc["offset_mapping"][0]          # (T,2)
        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]  # (T,)
        token_type_ids = enc["token_type_ids"][0] if "token_type_ids" in enc else None

        token_labels = torch.zeros(self.max_len, dtype=torch.float32)

        # real tokens: not padding AND not special tokens (specials often have offset (0,0))
        is_real_token = (attention_mask == 1) & (offsets[:, 1] > offsets[:, 0])

        spans = row["span_ranges"] if "span_ranges" in row.index else []
        if not isinstance(spans, list):
            spans = []

        # assign token label = 1 if token overlaps any annotated span
        for i, (start, end) in enumerate(offsets.tolist()):
            if not bool(is_real_token[i].item()):
                continue
            for s, e in spans:
                if start < e and end > s:
                    token_labels[i] = 1.0
                    break

        enc.pop("offset_mapping")

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_labels": token_labels,
            "token_loss_mask": is_real_token,
            "paragraph_label": torch.tensor(float(row["label_bin"]), dtype=torch.float32),
        }
        if token_type_ids is not None:
            batch["token_type_ids"] = token_type_ids

        return batch