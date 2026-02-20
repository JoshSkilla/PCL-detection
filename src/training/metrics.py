from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import numpy as np

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