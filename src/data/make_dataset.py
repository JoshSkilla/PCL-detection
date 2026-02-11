import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
SPLIT_DIR = Path("data/splits")

RAW_PCL_TASK1 = RAW_DIR / "dontpatronizeme_pcl.tsv"

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

if __name__ == "__main__":
    make_pcl_task1_dataset(save=True)