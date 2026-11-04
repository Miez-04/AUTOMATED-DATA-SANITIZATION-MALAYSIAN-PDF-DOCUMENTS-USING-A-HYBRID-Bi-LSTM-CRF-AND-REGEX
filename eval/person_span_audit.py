"""
pdf-sanitizer/eval/person_span_audit.py

Sorts the PERSON spans of one or more IOB2 dumps (the files eval.dump_iob2 or the
IOB2 inspector page produce) into the error groups found on the v29 dump:

  list marker   span contains a bare "b" / "B" token ("Objektif b. Panduan")
  fused         has a name connector but also an agency code, state or agency word
  no connector  no bin/binti/a/l/a/p/anak/@ (headings, short phrases, or Chinese names)
  clean         has a connector and nothing fused

Usage (Windows, from the project root):
    python -m eval.person_span_audit "dump v29.txt" "dump v30.txt"
"""
import argparse
import re

CONNECTORS = {"bin", "binti", "bt", "bte", "a/l", "a/p", "@", "s/o", "d/o", "anak", "ak"}
FUSED = re.compile(
    r"^(BPSH|BPK|BPPDP|BPL|BP|BSKK|BPSBP|BPKHAS|BSTP|BPI|BPLTV|BPM|BPG|BKP|BPA|PUU|UPK|SPP|PPD|SMK|JABATAN|"
    r"KEMENTERIAN|MASYARAKAT|PERAK|JOHOR|PAHANG|PERLIS|KEDAH|SELANGOR|MELAKA|SARAWAK|TERENGGANU|KELANTAN|"
    r"SABAH|SEMBILAN|PINANG|PULAU|LABUAN|PUTRAJAYA)$",
    re.IGNORECASE,
)


def norm(word):
    """Input: one token (str). Output: lower-case token without surrounding punctuation."""
    return word.strip(",.;:()").lower()


def person_spans(path):
    """
    Input : path (str) of an IOB2 dump, 'token TAG' per line
    Output: list of spans; each span is a list of token strings
    """
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line.strip():
                word, tag = line.rsplit(" ", 1)
                rows.append((word, tag))
    spans, i = [], 0
    while i < len(rows):
        if rows[i][1] == "B-PERSON":
            j = i + 1
            while j < len(rows) and rows[j][1] == "I-PERSON":
                j += 1
            spans.append([w for w, _ in rows[i:j]])
            i = j
        else:
            i += 1
    return spans


def classify(span):
    """Input: span (list of str). Output: group name (str)."""
    words = [norm(w) for w in span]
    if "b" in words:
        return "list marker"
    has_conn = any(w in CONNECTORS for w in words)
    if has_conn and any(FUSED.match(w.strip(",.")) for w in span):
        return "fused"
    return "clean" if has_conn else "no connector"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dumps", nargs="+")
    args = parser.parse_args()
    for path in args.dumps:
        try:
            spans = person_spans(path)
        except OSError as e:
            print(f"Cannot read {path}: {e}")
            continue
        groups = {}
        for s in spans:
            groups.setdefault(classify(s), []).append(" ".join(s))
        print(f"\n{path}\n  PERSON spans: {len(spans)}")
        for name in ("clean", "list marker", "fused", "no connector"):
            items = groups.get(name, [])
            print(f"  {name:13s} {len(items):4d}   e.g. {[x[:36] for x in items[:3]]}")


if __name__ == "__main__":
    main()
