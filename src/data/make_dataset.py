import pandas as pd
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np

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


# ---------------------------------------------------------------------
# Task1+Task2 builder with normalized/aggregated span_ranges + validation
# ---------------------------------------------------------------------

def _to_str_safe(x):
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x)


def build_task1_task2_with_spans(
    *,
    raw_task1_path,
    raw_task2_path,
    train_split_path,
    dev_split_path,
):
    """
    Returns (train_df, dev_df, pcl_df, spans_df_norm)

    train_df/dev_df contain:
      - text from Task1
      - label_0to4, label_bin
      - span_ranges: List[(start, finish)] in Task1 character offsets

    Normalization:
      - clamp offsets to [0, len(task1_text)]
      - finish = min(finish, start + len(span_text))
      - drop invalid spans (start >= finish)
      - aggregate per par_id
    """
    # Task1
    pcl_df = pd.read_csv(raw_task1_path, sep="\t", header=None, skiprows=4)
    pcl_df.columns = ["par_id", "art_id", "keyword", "country_code", "text", "label_0to4"]
    pcl_df["label_0to4"] = pcl_df["label_0to4"].astype(int)
    pcl_df["label_bin"] = (pcl_df["label_0to4"] >= 2).astype(int)
    pcl_df["text"] = pcl_df["text"].map(_to_str_safe)

    # splits
    train_ids = pd.read_csv(train_split_path)
    dev_ids = pd.read_csv(dev_split_path)
    if "par_id" not in train_ids.columns or "par_id" not in dev_ids.columns:
        raise ValueError("Split CSVs must contain a 'par_id' column")

    train_df = pcl_df.merge(train_ids[["par_id"]], on="par_id", how="inner")
    dev_df = pcl_df.merge(dev_ids[["par_id"]], on="par_id", how="inner")

    # Task2
    spans_df = pd.read_csv(raw_task2_path, sep="\t", header=None, skiprows=4)
    spans_df.columns = [
        "par_id",
        "art_id",
        "text",
        "keyword",
        "country_code",
        "span_start",
        "span_finish",
        "span_text",
        "category",
        "n_annotators",
    ]

    # normalize spans to Task1 text
    spans_df_norm = spans_df.dropna(subset=["span_start", "span_finish"]).copy()
    spans_df_norm["span_start"] = spans_df_norm["span_start"].astype(int)
    spans_df_norm["span_finish"] = spans_df_norm["span_finish"].astype(int)
    spans_df_norm["span_text"] = spans_df_norm["span_text"].map(_to_str_safe)

    len_by_par = pcl_df.set_index("par_id")["text"].map(len).to_dict()
    spans_df_norm["_L"] = spans_df_norm["par_id"].map(len_by_par)
    spans_df_norm = spans_df_norm[spans_df_norm["_L"].notna()].copy()
    spans_df_norm["_L"] = spans_df_norm["_L"].astype(int)

    spans_df_norm["span_start_norm"] = spans_df_norm.apply(
        lambda r: max(0, min(int(r["span_start"]), int(r["_L"]))), axis=1
    )
    spans_df_norm["span_finish_norm_raw"] = spans_df_norm.apply(
        lambda r: max(0, min(int(r["span_finish"]), int(r["_L"]))), axis=1
    )
    spans_df_norm["span_finish_norm"] = spans_df_norm.apply(
        lambda r: min(
            int(r["span_finish_norm_raw"]),
            int(r["span_start_norm"]) + len(r["span_text"] or ""),
        ),
        axis=1,
    )

    spans_df_norm = spans_df_norm[
        spans_df_norm["span_start_norm"] < spans_df_norm["span_finish_norm"]
    ].copy()

    span_ranges_by_par = (
        spans_df_norm.groupby("par_id")[["span_start_norm", "span_finish_norm"]]
        .apply(lambda d: list(map(tuple, d.to_numpy().tolist())))
        .rename("span_ranges")
        .reset_index()
    )

    train_df = (
        train_df.drop(columns=["span_ranges"], errors="ignore")
        .merge(span_ranges_by_par, on="par_id", how="left")
    )
    dev_df = (
        dev_df.drop(columns=["span_ranges"], errors="ignore")
        .merge(span_ranges_by_par, on="par_id", how="left")
    )

    train_df["span_ranges"] = train_df["span_ranges"].apply(lambda x: x if isinstance(x, list) else [])
    dev_df["span_ranges"] = dev_df["span_ranges"].apply(lambda x: x if isinstance(x, list) else [])

    return train_df, dev_df, pcl_df, spans_df_norm


def validate_span_ranges(
    df,
    *,
    pcl_df,
    spans_df_norm,
    name="df",
    assert_in_bounds=True,
    assert_pairs_exist=True,
    assert_negatives_empty=False,
):
    """
    Hard checks:
      A) df.span_ranges pairs exist in spans_df_norm for same par_id
      B) each (s,e) within [0, len(Task1 text)] and s < e
      C) (optional) label_bin==0 => empty span_ranges
    """
    for col in ["par_id", "text", "label_bin", "span_ranges"]:
        if col not in df.columns:
            raise ValueError(f"{name}: missing column '{col}'")

    t1_text = pcl_df.set_index("par_id")["text"].map(_to_str_safe).to_dict()
    lookup = (
        spans_df_norm.groupby("par_id")[["span_start_norm", "span_finish_norm"]]
        .apply(lambda d: set(map(tuple, d.to_numpy().tolist())))
        .to_dict()
    )

    def _norm_list(x):
        if not isinstance(x, list):
            return []
        out = []
        for t in x:
            if isinstance(t, (tuple, list)) and len(t) == 2:
                out.append((int(t[0]), int(t[1])))
        return out

    spans_col = df["span_ranges"].apply(_norm_list)

    missing_pairs = []
    if assert_pairs_exist:
        for pid, sp_list in zip(df["par_id"].tolist(), spans_col.tolist()):
            valid = lookup.get(int(pid), set())
            for s, e in sp_list:
                if (s, e) not in valid:
                    missing_pairs.append((int(pid), s, e))

    out_of_bounds = []
    if assert_in_bounds:
        for pid, sp_list in zip(df["par_id"].tolist(), spans_col.tolist()):
            L = len(t1_text.get(int(pid), ""))
            for s, e in sp_list:
                if s < 0 or e < 0 or s >= e or s > L or e > L:
                    out_of_bounds.append((int(pid), s, e, L))

    neg_with_spans = []
    if assert_negatives_empty:
        neg_mask = df["label_bin"].astype(int).eq(0)
        for pid, sp_list in zip(df.loc[neg_mask, "par_id"].tolist(), spans_col.loc[neg_mask].tolist()):
            if sp_list:
                neg_with_spans.append((int(pid), sp_list))

    print(f"\n== validate_span_ranges: {name} ==")
    print("shape:", df.shape)
    print("missing (par_id,s,e) pairs:", len(missing_pairs))
    if missing_pairs[:10]:
        print("examples:", missing_pairs[:10])
    print("out-of-bounds (par_id,s,e,L):", len(out_of_bounds))
    if out_of_bounds[:10]:
        print("examples:", out_of_bounds[:10])
    print("negatives with spans:", len(neg_with_spans))
    if neg_with_spans[:10]:
        print("examples:", neg_with_spans[:10])

    if assert_pairs_exist and missing_pairs:
        raise AssertionError("span_ranges contains pairs not present in spans_df_norm lookup")
    if assert_in_bounds and out_of_bounds:
        raise AssertionError("span_ranges contains out-of-bounds spans vs Task1 text")
    if assert_negatives_empty and neg_with_spans:
        raise AssertionError("Found label_bin==0 rows with non-empty span_ranges")


def find_span_text_mismatches(
    *,
    spans_df_norm,
    pcl_df,
    start_col: str = "span_start_norm",
    finish_col: str = "span_finish_norm",
    span_text_col: str = "span_text",
    par_id_col: str = "par_id",
    strip: bool = True,
):
    """
    Diagnostic only: returns rows where Task2 span_text != Task1 substring[start:finish].

    This is the check that used to show the "2 bad examples".
    It does NOT mean spans are unusable (bounds can still be correct).
    """
    import pandas as pd

    sp = spans_df_norm.copy()

    t1_by_par = pcl_df.set_index(par_id_col)["text"].map(_to_str_safe).to_dict()
    sp["_t1_text"] = sp[par_id_col].map(t1_by_par)

    sp["_t1_sub"] = sp.apply(
        lambda r: _to_str_safe(r["_t1_text"])[int(r[start_col]) : int(r[finish_col])],
        axis=1,
    )

    a = sp["_t1_sub"].map(_to_str_safe)
    b = sp[span_text_col].map(_to_str_safe)
    if strip:
        a = a.map(str.strip)
        b = b.map(str.strip)

    bad = sp.loc[a != b, [par_id_col, start_col, finish_col, span_text_col, "_t1_sub"]].copy()
    bad = bad.rename(columns={span_text_col: "span_text", "_t1_sub": "task1_substr"})
    return bad


def validate_span_text_alignment(
    df,
    *,
    pcl_df,
    spans_df_norm,
    name: str = "df",
    max_mismatches: int = 0,
    strip: bool = True,
) -> None:
    """
    Convenience wrapper: checks Task2 span_text vs Task1 substring for the *spans present in df.span_ranges*.

    Set max_mismatches>0 if you want to tolerate small annotation noise.
    """
    import pandas as pd

    if "par_id" not in df.columns or "span_ranges" not in df.columns:
        raise ValueError(f"{name}: df must contain 'par_id' and 'span_ranges'")

    # explode df.span_ranges -> (par_id, start, finish)
    rows = []
    for pid, ranges in zip(df["par_id"].tolist(), df["span_ranges"].tolist()):
        if not isinstance(ranges, list):
            continue
        for t in ranges:
            if isinstance(t, (tuple, list)) and len(t) == 2:
                rows.append((int(pid), int(t[0]), int(t[1])))

    pairs = pd.DataFrame(rows, columns=["par_id", "span_start_norm", "span_finish_norm"])
    if pairs.empty:
        print(f"\n== validate_span_text_alignment: {name} ==\n(no spans)")
        return

    # join span_text from spans_df_norm (normalized offsets)
    span_text = spans_df_norm[["par_id", "span_start_norm", "span_finish_norm", "span_text"]].copy()
    merged = pairs.merge(span_text, on=["par_id", "span_start_norm", "span_finish_norm"], how="left")

    missing_text = int(merged["span_text"].isna().sum())
    if missing_text:
        raise AssertionError(f"{name}: {missing_text} span_ranges pairs not found in spans_df_norm to fetch span_text")

    bad = find_span_text_mismatches(
        spans_df_norm=merged.rename(columns={"span_start_norm": "span_start_norm", "span_finish_norm": "span_finish_norm"}),
        pcl_df=pcl_df,
        start_col="span_start_norm",
        finish_col="span_finish_norm",
        span_text_col="span_text",
        par_id_col="par_id",
        strip=strip,
    )

    print(f"\n== validate_span_text_alignment: {name} ==")
    print("total spans checked:", len(merged))
    print("text mismatches:", len(bad))
    if len(bad):
        print("examples:")
        display(bad.head(20))

    if len(bad) > int(max_mismatches):
        raise AssertionError(f"{name}: too many span_text mismatches ({len(bad)} > {max_mismatches})")


def truncation_rate_by_label(tokenizer, df, max_len: int):
    lens = []
    labels = df["label_bin"].astype(int).to_numpy()
    for t in df["text"].fillna("").astype(str).tolist():
        enc = tokenizer(t, truncation=True, max_length=max_len)
        lens.append(len(enc["input_ids"]))
    lens = np.array(lens)

    truncated = lens >= max_len
    pos = labels == 1
    neg = labels == 0

    return {
        "max_len": max_len,
        "all_truncated_pct": float(truncated.mean() * 100),
        "pos_truncated_pct": float(truncated[pos].mean() * 100) if pos.any() else 0.0,
        "neg_truncated_pct": float(truncated[neg].mean() * 100) if neg.any() else 0.0,
        "pos_p99_len": int(np.percentile(lens[pos], 99)) if pos.any() else None,
        "neg_p99_len": int(np.percentile(lens[neg], 99)) if neg.any() else None,
    }

if __name__ == "__main__":
    make_pcl_task1_dataset(save=True)
    # merge_pcl_task2_dataset(save=True)
