import argparse, re, sys
from pathlib import Path
from typing import Callable, Sequence, Tuple, List
from tqdm import tqdm

import numpy as np
from sklearn.utils import resample
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score)


def parse_lines(raw_lines: Sequence[str]) -> Tuple[List[int], List[int]]:
    """Extract ground truth and predictions from tab-separated lines."""
    true, pred = [], []
    pat = re.compile(r"(.+)\s+(\d)\s*$")
    for ln in raw_lines:
        m = pat.match(ln)
        if not m:
            continue
        path, p = m.groups()
        true.append(0 if "Normal" in path else 1)
        pred.append(int(p))
        assert int(p)==0 or int(p)==1, str(ln)
    return true, pred


def bootstrap_ci(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    metric: Callable[[Sequence[int], Sequence[int]], float],
    n: int = 1_000,
    alpha: float = 0.05,
    rng_seed: int = 42,
) -> Tuple[float, float]:

    rng = np.random.RandomState(rng_seed)
    pairs = list(zip(y_true, y_pred))
    vals = []

    for _ in tqdm(range(n)):
        sample = resample(pairs, n_samples=len(pairs), random_state=rng)
        yt, yp = zip(*sample)
        try:
            v = metric(yt, yp)
            if not np.isnan(v):
                vals.append(v)
        except ValueError:
            continue

    if not vals:
        return float("nan"), float("nan")

    lower = np.percentile(vals, 100 * alpha / 2)
    upper = np.percentile(vals, 100 * (1 - alpha / 2))
    return lower, upper



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", )
    ap.add_argument("-n", "--bootstraps", type=int, default=1_000,)
    args = ap.parse_args()

    raw = (Path(args.file).read_text(encoding="utf-8").splitlines()
           if args.file else
           sys.stdin.read().splitlines())

    y_true, y_pred = parse_lines(raw)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred,  zero_division=0)
    f1   = f1_score(y_true, y_pred,      zero_division=0)
    try:
        auc_roc = roc_auc_score(y_true, y_pred)
        pr_auc  = average_precision_score(y_true, y_pred)
    except: 
        auc_roc, pr_auc = -1, -1
    

    # ci_acc  = bootstrap_ci(y_true, y_pred, accuracy_score, args.bootstraps)
    # ci_prec = bootstrap_ci(y_true, y_pred,
    #                        lambda yt, yp: precision_score(yt, yp, zero_division=0),
    #                        args.bootstraps)
    # ci_rec  = bootstrap_ci(y_true, y_pred,
    #                        lambda yt, yp: recall_score(yt, yp, zero_division=0),
    #                        args.bootstraps)
    # ci_f1   = bootstrap_ci(y_true, y_pred,
    #                        lambda yt, yp: f1_score(yt, yp, zero_division=0),
    #                        args.bootstraps)
    # ci_auc  = bootstrap_ci(y_true, y_pred, roc_auc_score, args.bootstraps)
    # ci_pr   = bootstrap_ci(y_true, y_pred, average_precision_score,
    #                        args.bootstraps)

    def hw(ci):          # half-width helper
        return 0

    print(f"Samples         : {len(y_true)}")
    print(f"Accuracy        : {acc:.4f} ")
    print(f"Precision       : {prec:.4f}  ")
    print(f"Recall          : {rec:.4f} ")
    print(f"F1-score        : {f1:.4f}  ")
    print(f"AUC-ROC         : {auc_roc:.4f} ")
    print(f"PR-AUC          : {pr_auc:.4f}  ")
    print(f"{acc:.4f}  &{prec:.4f}  &{rec:.4f} &{f1:.4f} &{auc_roc:.4f} ")


if __name__ == "__main__":
    main()
