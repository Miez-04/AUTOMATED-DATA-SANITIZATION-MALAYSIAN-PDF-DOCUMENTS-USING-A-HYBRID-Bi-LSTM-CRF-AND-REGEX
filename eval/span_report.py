# pdf-sanitizer/eval/span_report.py
import argparse
from collections import Counter


def load(path):
    """
    Input : path (str), IOB2 dump with 'token TAG' per line, blank line = separator
    Output: list of (token str, tag str); separators become ('', 'SEP')
    """
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line.strip():
                out.append(("", "SEP"))
                continue
            w, t = line.rsplit(" ", 1)               # token and tag
            out.append((w, t))
    return out


def spans(toks):
    """
    Input : toks (list of (str, str))
    Output: list of (entity str, start int, end int, words list of str); end exclusive
    """
    out, i = [], 0
    while i < len(toks):
        if toks[i][1].startswith("B-"):
            ty, j = toks[i][1][2:], i + 1
            while j < len(toks) and toks[j][1] == "I-" + ty:
                j += 1
            out.append((ty, i, j, [w for w, _ in toks[i:j]]))
            i = j
        else:
            i += 1
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", nargs="+", help="one or more IOB2 dump files, one per model")
    args = parser.parse_args()
    for path in args.dump:
        try:
            toks = load(path)
        except OSError as e:
            print(f"Cannot read {path}: {e}")
            continue
        sp = spans(toks)
        n_sp = Counter(s[0] for s in sp)                          # spans per entity type
        n_tok = Counter(s[0] for s in sp for _ in range(s[2] - s[1]))   # tokens per entity type
        print(f"\n{path}\n  spans {dict(n_sp)}\n  tokens {dict(n_tok)}")
        for ty in ("PERSON", "ADDRESS"):
            print(f"  {ty}:")
            for s in [x for x in sp if x[0] == ty][:40]:
                print("    ", " ".join(s[3])[:110])


if __name__ == "__main__":
    main()