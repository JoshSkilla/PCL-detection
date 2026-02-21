import re

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