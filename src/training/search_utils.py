import os
import shutil
import pandas as pd
import optuna
from optuna.trial import TrialState
from transformers import TrainerCallback
from IPython.display import display
import pandas as pd
from typing import Any, Mapping, Optional, Sequence

# ::::::::::::::::::::::::::::::::::::::::::::::::::: Grid search utils :::::::::::::::::::::::::::::::::::::::::::::::::::


# ::::::::::::: Database related utils ::::::::::::::

def progress_df(study: optuna.Study) -> pd.DataFrame:
    """
    Returns a nice DataFrame of trials so far (params + value + state + user attrs).
    """
    df = study.trials_dataframe(attrs=("number", "state", "value", "params", "user_attrs"))
    # Tidy: flatten param columns a bit (Optuna already does this nicely)
    return df.sort_values(["state", "value"], ascending=[True, False])

def best_so_far_df(study: optuna.Study, n: int = 10) -> pd.DataFrame:
    df = study.trials_dataframe(attrs=("number", "state", "value", "params", "user_attrs"))
    df = df[df["state"] == "COMPLETE"].sort_values("value", ascending=False).head(n)
    return df


def reset_study_completely(study_db_path: str, output_dir: str) -> None:
    """
    Hard reset: deletes the SQLite DB file and trial run folders.
    This truly resets the sampler/pruner history.
    """
    if os.path.exists(study_db_path):
        os.remove(study_db_path)
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)


def clean_trial_folders(output_dir, study: optuna.Study, keep_states=(TrialState.COMPLETE,)) -> int:
    """
    Deletes trial_* folders for trials not in keep_states.
    Safe if you don't need failed/pruned checkpoints.
    """
    out = str(output_dir)
    keep = set()
    for t in study.get_trials(deepcopy=False):
        if t.state in keep_states:
            keep.add(f"trial_{t.number:04d}")

    removed = 0
    if not os.path.isdir(out):
        return 0

    for name in os.listdir(out):
        if name.startswith("trial_") and name not in keep:
            path = os.path.join(out, name)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
    return removed

def mark_stale_running_trials_as_fail(study: optuna.Study, reason: str = "stale_after_interrupt") -> int:
    """
    Convert any RUNNING trials (leftover after kernel stop/sleep/interrupt) to FAIL.
    Keeps the study consistent for resume.
    """
    n = 0
    for t in study.get_trials(deepcopy=False):
        if t.state == TrialState.RUNNING:
            # best-effort annotation (internal API)
            try:
                study._storage.set_trial_user_attr(t._trial_id, "stale_reason", reason)
            except Exception:
                pass
            study._storage.set_trial_state_values(t._trial_id, TrialState.FAIL, values=None)
            n += 1
    return n

COUNT_AS_DONE = (TrialState.COMPLETE, TrialState.PRUNED)

def remaining_trials_to_run(study, target_total_done: int, done_states=COUNT_AS_DONE) -> int:
    done = len(study.get_trials(states=done_states, deepcopy=False))
    return max(0, target_total_done - done)

def done_counts(study) -> dict:
    trials = study.get_trials(deepcopy=False)
    out = {s.name: 0 for s in TrialState}
    for t in trials:
        out[t.state.name] += 1
    return out



# ::::::::::::: Optuna related utils ::::::::::::::

class OptunaMedianPruningCallback(TrainerCallback):
    """
    Reports eval metric to Optuna each evaluation and prunes using the Study's pruner (MedianPruner).
    Requires:
      - eval_strategy != "no" (you use "epoch", good)
      - compute_metrics returns 'f1' so Trainer logs 'eval_f1' (your code expects metrics["eval_f1"])
    """
    def __init__(self, trial: optuna.Trial, monitor: str = "eval_f1"):
        self.trial = trial
        self.monitor = monitor

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return control
        if self.monitor not in metrics:
            return control

        value = metrics[self.monitor]

        # step: use epoch index when available (since you evaluate per epoch)
        if state.epoch is None:
            step = int(state.global_step)
        else:
            # epoch can be float (e.g., 1.0); convert to stable int step
            step = int(round(state.epoch))

        self.trial.report(value, step=step)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"MedianPruner: pruned at epoch={step} with {self.monitor}={value}")

        return control



# :::::::::::: Display utils ::::::::::::::

def pretty_print_dict(
    title: str,
    d: Mapping[str, Any],
    *,
    sort_keys: bool = True,
    key_order: Optional[Sequence[str]] = None,
    float_dp: int = 6,
):
    """
    Notebook-friendly pretty print for dict-like objects.
    Renders a 2-col table (key/value) in Jupyter; falls back to aligned text elsewhere.
    """

    def _fmt(v: Any) -> Any:
        # compact floats
        if isinstance(v, float):
            return round(v, float_dp)
        return v

    items = list((d or {}).items())

    if key_order:
        order = {k: i for i, k in enumerate(key_order)}
        items.sort(key=lambda kv: (order.get(kv[0], 10**9), kv[0] if sort_keys else order.get(kv[0], 0)))
    elif sort_keys:
        items.sort(key=lambda kv: kv[0])

    df = pd.DataFrame([(k, _fmt(v)) for k, v in items], columns=["key", "value"])

    # Prefer rich display in notebooks
    try:
        display(df.style.hide(axis="index").set_caption(title))
    except Exception:
        # Fallback: aligned text
        print(title)
        w = max([len(str(k)) for k in df["key"]] + [3])
        for k, v in df.itertuples(index=False):
            print(f"  {str(k):<{w}} : {v}")