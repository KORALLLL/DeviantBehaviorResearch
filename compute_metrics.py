import argparse, re, sys
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             )

def parse_lines(raw_lines):
    true, pred = [], []
    pat = re.compile(r"(.+)\t(\d)\s*$")
    for ln in raw_lines:
        m = pat.match(ln)
        if not m:
            continue
        path, p = m.groups()
        true.append(0 if "Normal" in path else 1)
        pred.append(int(p))
    return true, pred

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", help="txt-файл с результатами")
    args = ap.parse_args()

    raw = Path(args.file).read_text(encoding="utf-8").splitlines() \
          if args.file else sys.stdin.read().splitlines()

    y_true, y_pred = parse_lines(raw)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    auc_roc = roc_auc_score(y_true, y_pred)                 # [1]
    pr_auc  = average_precision_score(y_true, y_pred)       # [2]


    print(f"Samples         : {len(y_true)}")
    print(f"Accuracy        : {acc:.4f}")
    print(f"Precision       : {prec:.4f}")
    print(f"Recall          : {rec:.4f}")
    print(f"F1-score        : {f1:.4f}")
    print(f"AUC-ROC         : {auc_roc:.4f}")
    print(f"PR-AUC          : {pr_auc:.4f}")

if __name__ == "__main__":
    main()
