import pandas as pd
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]   # src/data -> src -> project root

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SPLIT_DIR = ROOT / "data" / "splits"

RAW_PCL_TASK1 = RAW_DIR / "dontpatronizeme_pcl.tsv"
RAW_PCL_TASK2 = RAW_DIR / "dontpatronizeme_categories.tsv"

TRAIN_SPLIT = SPLIT_DIR / "train_semeval_parids-labels.csv"
DEV_SPLIT = SPLIT_DIR / "dev_semeval_parids-labels.csv"

def make_pcl_task1_dataset(save=True):
    pcl_df  = pd.read_csv(
        RAW_PCL_TASK1,
        sep="\t",
        header=None,
        skiprows=4
        )
    pcl_df.columns = [
        "par_id",
        "art_id",
        "keyword",
        "country_code",
        "text",
        "label_0to4",
    ]

    # For binary classification
    pcl_df["label_bin"] = (pcl_df["label_0to4"] >= 2).astype(int)

    # Get given splits
    train_ids = pd.read_csv(TRAIN_SPLIT)
    dev_ids = pd.read_csv(DEV_SPLIT)

    train_df = pcl_df.merge(train_ids[["par_id"]], how="inner", on="par_id")
    dev_df = pcl_df.merge(dev_ids[["par_id"]], how="inner", on="par_id")

    if save:
        train_df.to_csv(PROCESSED_DIR / "pcl_task1_train.csv", index=False)
        dev_df.to_csv(PROCESSED_DIR / "pcl_task1_dev.csv", index=False)
        print(f"Saved processed datasets to {PROCESSED_DIR}")
    return train_df, dev_df


def merge_pcl_task2_dataset(save=True):
    """
    FULL merge:
    - keeps ALL Task1 paragraphs (label_bin in {0,1})
    - adds Task2 multi-label category columns where available (span-level -> paragraph-level)
    - non-PCL rows get all-zero category vector
    - applies official train/dev splits
    """

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load task1 raw
    pcl_df = pd.read_csv(RAW_PCL_TASK1, sep="\t", header=None, skiprows=4)
    pcl_df.columns = ["par_id", "art_id", "keyword", "country_code", "text", "label_0to4"]
    pcl_df["label_bin"] = (pcl_df["label_0to4"] >= 2).astype(int)

    # Load task2 raw
    cat_df = pd.read_csv(RAW_PCL_TASK2, sep="\t", header=None, skiprows=4)
    cat_df.columns = [
        "par_id", "art_id", "text", "keyword", "country_code",
        "span_start", "span_finish", "span_text", "category", "n_annotators"
    ]
    cat_df = cat_df[["par_id", "category"]]  # only what we need

    # get label set
    pcl_categories = sorted(cat_df["category"].dropna().unique().tolist())

    # paragraph-level label sets
    cat_grouped = (
        cat_df.groupby("par_id")["category"]
        .apply(lambda x: sorted(set(x)))
        .reset_index()
    )

    # multi-hot encoding
    for c in pcl_categories:
        cat_grouped[c] = cat_grouped["category"].apply(lambda labels: int(c in labels))
    cat_grouped = cat_grouped.drop(columns=["category"])

    # merge paragraphs with all detected category labels
    full_df = pcl_df.merge(cat_grouped, on="par_id", how="left")
    for c in pcl_categories:
        full_df[c] = full_df[c].fillna(0).astype(int)

    # apply splits
    train_ids = pd.read_csv(TRAIN_SPLIT)
    dev_ids = pd.read_csv(DEV_SPLIT)

    train_df = full_df.merge(train_ids[["par_id"]], on="par_id", how="inner")
    dev_df   = full_df.merge(dev_ids[["par_id"]], on="par_id", how="inner")

    if save:
        train_df.to_csv(PROCESSED_DIR / "pcl_task2_train_all.csv", index=False)
        dev_df.to_csv(PROCESSED_DIR / "pcl_task2_dev_all.csv", index=False)
        print(f"Saved FULL Task-2 merged datasets to: {PROCESSED_DIR}")
        print("Detected Task-2 labels:", pcl_categories)

    return train_df, dev_df

def make_pcl_task2_dataset(save: bool = True):
    """
    Export Task 2 at SPAN level (one row per annotated span).

    This keeps the evidence spans (span_text, offsets, annotator agreement) and
    adds one-hot columns for the PCL strategy category.
    """

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load task2 raw
    spans_df = pd.read_csv(RAW_PCL_TASK2, sep="\t", header=None, skiprows=4)
    spans_df.columns = [
        "par_id", "art_id", "text", "keyword", "country_code",
        "span_start", "span_finish", "span_text", "category", "n_annotators"
    ]

    # Detect label set 
    pcl_categories = sorted(spans_df["category"].dropna().unique().tolist())

    # One-hot encoding per span
    for c in pcl_categories:
        spans_df[c] = (spans_df["category"] == c).astype(int)

    if save:
        out_path = PROCESSED_DIR / "pcl_task2_spans.csv"
        spans_df.to_csv(out_path, index=False)
        print(f"Saved Task-2 span-level dataset to: {out_path}")
        print("Detected Task-2 labels:", pcl_categories)

    return spans_df


def clean_text(t):
    t = re.sub(r"&\w+;", " ", t)
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


if __name__ == "__main__":
    make_pcl_task1_dataset(save=True)
    # merge_pcl_task2_dataset(save=True)
