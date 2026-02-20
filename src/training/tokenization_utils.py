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
        return dict(tokenized)

    # Tokenize datasets & truncate/padd them to max length
    ds_train = ds_train_raw.map(tok_fn, batched=True)
    ds_val   = ds_val_raw.map(tok_fn, batched=True)
    ds_dev   = ds_dev_raw.map(tok_fn, batched=True)

    ds_train = ds_train.rename_column("label_bin", "labels")
    ds_train = ds_train.remove_columns([c for c in ds_train.column_names if c not in ["input_ids", "attention_mask", "labels"]])

    ds_val = ds_val.rename_column("label_bin", "labels")
    ds_val = ds_val.remove_columns([c for c in ds_val.column_names if c not in ["input_ids", "attention_mask", "labels"]])

    ds_dev = ds_dev.rename_column("label_bin", "labels")
    ds_dev = ds_dev.remove_columns([c for c in ds_dev.column_names if c not in ["input_ids", "attention_mask", "labels"]])

    return tok, ds_train, ds_val, ds_dev