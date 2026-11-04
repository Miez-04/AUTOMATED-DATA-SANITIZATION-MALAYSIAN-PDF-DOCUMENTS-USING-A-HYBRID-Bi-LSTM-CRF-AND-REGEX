"""
pdf-sanitizer/eval/compare_to_gold.py

Lists every gold span the model missed and every span it invented, with context,
and summarises per page type. Use it on the IOB2 dump the web app's inspector page
(or eval.dump_iob2) makes for the benchmark PDF, together with the gold file from
eval.build_benchmark_gold. Both files must hold the same tokens in the same order.

Usage (Windows, one line):
    python -m eval.compare_to_gold data/processed/real_benchmark_50_gold.txt "<dump file>" --types PERSON ADDRESS
"""
import argparse
from collections import Counter, defaultdict


def read_pages(path):
    """
    Input : path (str), 'token TAG' per line, blank line between pages (or none)
    Output: list of (token, tag) tuples for the whole file, and the page break positions
    """
    if path.lower().endswith(".pdf"):
        raise ValueError(f"{path} is a PDF. This script needs the IOB2 text dump the web app's inspector page "
                         f"makes for that PDF (lines of 'token TAG'), not the PDF itself.")
    rows, breaks = [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for number, line in enumerate(f, start=1):
            line = line.rstrip("\r\n")
            if not line.strip():
                if rows and (not breaks or breaks[-1] != len(rows)):
                    breaks.append(len(rows))
                continue
            parts = line.rsplit(" ", 1)
            if len(parts) != 2 or not (parts[1] == "O" or parts[1][:2] in ("B-", "I-")):
                raise ValueError(f"{path}, line {number} is not 'token TAG' (found: {line[:60]!r}). "
                                 f"Use the IOB2 dump from the inspector page.")
            rows.append((parts[0], parts[1]))
    return rows, breaks


def spans_of(rows):
    """Input: list of (token, tag). Output: dict {(type, start, end): text} with end exclusive."""
    out, i = {}, 0
    while i < len(rows):
        if rows[i][1].startswith("B-"):
            etype, j = rows[i][1][2:], i + 1
            while j < len(rows) and rows[j][1] == "I-" + etype:
                j += 1
            out[(etype, i, j)] = " ".join(t for t, _ in rows[i:j])
            i = j
        else:
            i += 1
    return out


def page_type(words):
    """Input: tokens of one page. Output: layout name (str), from a few header words."""
    text = " ".join(words).upper()
    for key, name in (("UTILITY", "utility bill"), ("POLIS DIRAJA", "police report"), ("TENANCY", "tenancy"),
                      ("EMPLOYMENT", "employment letter"), ("TAX INVOICE", "invoice")):
        if key in text:
            return name
    return "other"


def context(rows, start, end, width=4):
    left = " ".join(t for t, _ in rows[max(0, start - width):start])
    right = " ".join(t for t, _ in rows[end:end + width])
    return f"{left} [[{' '.join(t for t, _ in rows[start:end])}]] {right}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gold")
    parser.add_argument("dump")
    parser.add_argument("--types", nargs="+", default=["PERSON", "ADDRESS"])
    parser.add_argument("--limit", type=int, default=25, help="examples printed per list")
    args = parser.parse_args()

    try:
        gold_rows, breaks = read_pages(args.gold)
        pred_rows, _ = read_pages(args.dump)
    except (OSError, ValueError) as e:
        print(f"Cannot use input: {e}")
        return
    if [t for t, _ in gold_rows] != [t for t, _ in pred_rows]:
        n = next((i for i, (a, b) in enumerate(zip(gold_rows, pred_rows)) if a[0] != b[0]), min(len(gold_rows), len(pred_rows)))
        print(f"Token streams differ (gold {len(gold_rows)} tokens, dump {len(pred_rows)}); first difference at token {n}: "
              f"{gold_rows[n][0] if n < len(gold_rows) else '-'!r} vs {pred_rows[n][0] if n < len(pred_rows) else '-'!r}")
        return

    gold, pred = spans_of(gold_rows), spans_of(pred_rows)
    bounds = [0] + breaks + [len(gold_rows)]
    page_of = lambda pos: next(k for k in range(len(bounds) - 1) if bounds[k] <= pos < bounds[k + 1])
    layouts = [page_type([t for t, _ in gold_rows[bounds[k]:bounds[k + 1]]]) for k in range(len(bounds) - 1)]

    for etype in args.types:
        g = {k: v for k, v in gold.items() if k[0] == etype}
        p = {k: v for k, v in pred.items() if k[0] == etype}
        tp = set(g) & set(p)
        fn, fp = set(g) - tp, set(p) - tp
        print(f"\n===== {etype}: gold {len(g)} | predicted {len(p)} | exact matches {len(tp)} | missed {len(fn)} | false {len(fp)} =====")

        by_layout = defaultdict(Counter)
        for k in g:
            by_layout[layouts[page_of(k[1])]]["gold"] += 1
            if k in tp:
                by_layout[layouts[page_of(k[1])]]["found"] += 1
        for k in fp:
            by_layout[layouts[page_of(k[1])]]["false"] += 1
        for name, c in sorted(by_layout.items()):
            print(f"  {name:18s} gold {c['gold']:3d}  found {c['found']:3d}  false {c['false']:3d}")

        print(f"\n  MISSED (first {args.limit}):")
        for k in sorted(fn, key=lambda x: x[1])[:args.limit]:
            overlap = [q for q in pred if q[1] < k[2] and k[1] < q[2]]
            kind = "missed entirely" if not overlap else ("different boundaries" if any(q[0] == etype for q in overlap)
                                                         else "labelled " + overlap[0][0])
            print(f"    p{page_of(k[1]) + 1:<3d} {kind:22s} {context(gold_rows, k[1], k[2])}")
        print(f"\n  FALSE (first {args.limit}):")
        for k in sorted(fp, key=lambda x: x[1])[:args.limit]:
            overlap = [q for q in gold if q[1] < k[2] and k[1] < q[2]]
            kind = "no gold span" if not overlap else ("different boundaries" if any(q[0] == etype for q in overlap)
                                                      else "gold is " + overlap[0][0])
            print(f"    p{page_of(k[1]) + 1:<3d} {kind:22s} {context(pred_rows, k[1], k[2])}")


if __name__ == "__main__":
    main()
