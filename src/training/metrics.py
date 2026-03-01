from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import numpy as np
import torch

def compute_metrics_from_logits(logits, labels, threshold=0.5):
    # logits: (N,1) or (N,)
    logits = np.squeeze(logits)
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)

    y = labels.astype(int)

    f1  = f1_score(y, preds, zero_division=0)
    p   = precision_score(y, preds, zero_division=0)
    r   = recall_score(y, preds, zero_division=0)
    acc = accuracy_score(y, preds)

    cm = confusion_matrix(y, preds, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    acc0 = tn / (tn + fp + 1e-12)
    acc1 = tp / (tp + fn + 1e-12)

    return {
        "f1": f1,
        "precision": p,
        "recall": r,
        "accuracy": acc,
        "acc_nonpcl": acc0,
        "acc_pcl": acc1,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return compute_metrics_from_logits(logits, labels, threshold=0.5)


# :::::::: Custom metrics for localised PCL detection training loop  :::::::::

def _bin_metrics_from_logits(logits: np.ndarray, labels: np.ndarray, threshold: float = 0.5):
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)
    y = labels.astype(int)

    tp = int(((preds == 1) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    acc_nonpcl = tn / max(tn + fp, 1)   # specificity
    acc_pcl    = tp / max(tp + fn, 1)   # recall for positives

    return {
        "acc": float(acc),
        "acc_nonpcl": float(acc_nonpcl),
        "acc_pcl": float(acc_pcl),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }

def _token_metrics_from_logits(
    token_logits: torch.Tensor,
    token_labels: torch.Tensor,
    token_mask: torch.Tensor,
    logit_thresh: float = 0.0,  # 0.0 == prob 0.5
):
    """
    token_logits: (B,T)
    token_labels: (B,T) in {0,1}
    token_mask:   (B,T) bool (True = valid token)
    Returns token-level acc, acc0, acc1, tp/tn/fp/fn.
    """
    m = token_mask.to(torch.bool)
    if m.sum().item() == 0:
        return {"tok_acc": None, "tok_acc0": None, "tok_acc1": None, "tp": 0, "tn": 0, "fp": 0, "fn": 0}

    logits = token_logits[m].detach().float().cpu()
    y = token_labels[m].detach().float().cpu()
    preds = (logits >= logit_thresh).to(torch.int32).numpy()
    y = (y >= 0.5).to(torch.int32).numpy()

    tp = int(((preds == 1) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())

    tok_acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    tok_acc0 = tn / max(tn + fp, 1)
    tok_acc1 = tp / max(tp + fn, 1)

    return {"tok_acc": float(tok_acc), "tok_acc0": float(tok_acc0), "tok_acc1": float(tok_acc1), "tp": tp, "tn": tn, "fp": fp, "fn": fn}

def stats_on_loader(
    loader,
    model,
    DEVICE,
    USE_AMP,
    AMP_DTYPE,
    threshold: float = 0.5,
    compute_token_loss: bool = False,   # <- default OFF for speed
    compute_token_metrics: bool = False,
    token_logit_thresh: float = 0.0,
    limit_batches: int | None = None,
):
    model.eval()
    all_logits, all_labels = [], []
    losses, losses_par, losses_tok = [], [], []

    # token metrics accumulators
    tok_tp = tok_tn = tok_fp = tok_fn = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if limit_batches is not None and i >= limit_batches:
                break

            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            # Skip token loss (expensive) unless explicitly requested
            if not compute_token_loss:
                batch = dict(batch)
                batch["token_labels"] = None  # keeps token_loss_mask for pooling, but disables BCE token loss

            if USE_AMP:
                with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                    out = model(**batch)
            else:
                out = model(**batch)

            all_logits.append(out["paragraph_logit"].detach().float().cpu())
            all_labels.append(batch["paragraph_label"].detach().float().cpu())

            if out.get("loss") is not None:
                losses.append(float(out["loss"].detach().float().cpu()))
            if out.get("loss_par") is not None:
                losses_par.append(float(out["loss_par"].detach().float().cpu()))
            if out.get("loss_tok") is not None:
                losses_tok.append(float(out["loss_tok"].detach().float().cpu()))

            # Optional token metrics (also costs extra; keep it limited)
            if compute_token_metrics and batch.get("token_labels") is not None:
                tm = _token_metrics_from_logits(
                    token_logits=out["token_logits"],
                    token_labels=batch["token_labels"],
                    token_mask=batch["token_loss_mask"],
                    logit_thresh=token_logit_thresh,
                )
                tok_tp += tm["tp"]; tok_tn += tm["tn"]; tok_fp += tm["fp"]; tok_fn += tm["fn"]

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)

    m = _bin_metrics_from_logits(logits, labels, threshold=threshold)
    m["f1"] = float(f1_score(labels, (1.0 / (1.0 + np.exp(-logits)) >= threshold).astype(int), pos_label=1))

    m["loss"] = float(np.mean(losses)) if losses else None
    m["loss_par"] = float(np.mean(losses_par)) if losses_par else None
    m["loss_tok"] = float(np.mean(losses_tok)) if losses_tok else None

    if compute_token_metrics:
        m["tok_acc"] = float((tok_tp + tok_tn) / max(tok_tp + tok_tn + tok_fp + tok_fn, 1))
        m["tok_acc0"] = float(tok_tn / max(tok_tn + tok_fp, 1))
        m["tok_acc1"] = float(tok_tp / max(tok_tp + tok_fn, 1))

    return m