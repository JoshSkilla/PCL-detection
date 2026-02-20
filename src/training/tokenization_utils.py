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