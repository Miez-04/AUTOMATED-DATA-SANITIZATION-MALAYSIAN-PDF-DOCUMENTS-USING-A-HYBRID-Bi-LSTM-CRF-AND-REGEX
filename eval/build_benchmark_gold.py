"""
pdf-sanitizer/eval/build_benchmark_gold.py

Builds gold IOB2 labels for malaysian_pii_50_strict_format_randomized.pdf, using
the SAME tokens the web app extracts (pdf_ingestion.PDFIngestor), so the file can be
scored directly with eval.run_eval against any checkpoint.

Gold policy (change with the flags if you decide differently):
    PERSON   every occurrence of the data subject's name (body text, signature line, "Dear ...")
    NRIC     every MyKad number in the three layouts (6-2-4 hyphenated, spaced hyphens, 12 digits)
    PHONE    the data subject's own number. Organisation lines ("Tel:", "Customer Careline:",
             "careline", "HR line at") are NOT PII unless --org-phones is given.
    ADDRESS  the data subject's address. Letterhead addresses of the organisation are NOT PII
             unless --org-addresses is given.

Usage (from the project root, Windows):
    python -m eval.build_benchmark_gold "<path to the PDF>" --out data/processed/real_benchmark_50_gold.txt
"""
import argparse
import re
import sys
from pathlib import Path

NRIC = re.compile(r"(?<!\d)\d{6}\s*-?\s*\d{2}\s*-?\s*\d{4}(?!\d)")
PHONE_CANDIDATE = re.compile(r"(?<![\w/\-.,])\+?\d[\d\- ]{6,16}\d(?![\w/\-.,]\d)")
ORG_PHONE_CONTEXT = re.compile(r"(?:Careline:|careline|Tel:|Tel |HR line at)\s*$", re.IGNORECASE)

# Where the data subject's name appears: (regex with one capture group)
NAME_CONTEXTS = [
    r"Nama Pelanggan: (.+?) No\. MyKad:",
    r"Nama Pengadu: (.+?) No\. MyKad:",
    r"TENANT FULL NAME: (.+?) NATIONAL ID",
    r"Full Name: (.+?) MyKad / NRIC No:",
    r"certifies that (.+?) \(identity card number",
    r"Dear (.+?),",
    r"pemegang akaun berdaftar (.+?) \(No\. Kad Pengenalan",
    r"lease of residential property to (.+?), holding MyKad",
    r"issued to (.+?), holding MyKad",
    r"pengadu bernama (.+?) yang memegang",
]
# Where the data subject's address appears
ADDRESS_CONTEXTS = [
    r"Alamat Premis: (.+?) JUMLAH BIL:",
    r"Alamat: (.+?) Laporan polis ini",
    r"premis di (.+?) dengan jumlah bil",
    r"recorded at (.+?)\s*(?<=[a-z])\.? All official",
    r"Designated Address: (.+?) Contact Number:",
    r"Tenant residing at (.+?) agrees to the",
    r"DEMISED PROPERTY ADDRESS: (.+?) The parties",
    r"Billing Address: (.+?) ACCOUNT SUMMARY:",
    r"dispatched to (.+?)\. For payment",
    r"beralamat di (.+?) telah hadir",
]
# Organisation letterhead address: line that starts a page or follows the company name
ORG_ADDRESS = re.compile(r"((?:Level|No\.|Menara|Tower|Main Block|All Asia|South Support|RedQ|\d+ Jalan|5 Jalan)[^|]*?\d{5}[^|]*?)(?= \| |\s(?:Tel:|TAX INVOICE|CONFIDENTIAL EMPLOYMENT|STATEMENT OF|Customer))")


def page_text_and_offsets(tokens):
    """Input: tokens (list of str). Output: (text, [(start, end) per token]) with single spaces between tokens."""
    offsets, pos, parts = [], 0, []
    for tok in tokens:
        offsets.append((pos, pos + len(tok)))
        parts.append(tok)
        pos += len(tok) + 1
    return " ".join(parts), offsets


def char_spans(text, tokens_offsets, regexes, group=1):
    """Input: page text, regex list. Output: list of (start, end) character spans of the capture group."""
    out = []
    for pattern in regexes:
        for m in re.finditer(pattern, text):
            out.append((m.start(group), m.end(group)))
    return out


def label_tokens(offsets, spans_by_type):
    """Input: token offsets, {type: [(start, end)]}. Output: list of IOB2 tags, one per token."""
    tags = ["O"] * len(offsets)
    for etype, spans in spans_by_type.items():
        for s, e in spans:
            first = True
            for i, (ts, te) in enumerate(offsets):
                if ts < e and te > s:           # token overlaps the span
                    if tags[i] == "O":
                        tags[i] = ("B-" if first else "I-") + etype
                    first = False
    return tags


def gold_for_page(tokens, org_phones=False, org_addresses=False):
    text, offsets = page_text_and_offsets(tokens)
    spans = {"PERSON": [], "NRIC": [], "PHONE": [], "ADDRESS": []}

    names = {text[s:e].strip() for s, e in char_spans(text, offsets, NAME_CONTEXTS)}
    for name in names:                           # every occurrence of each harvested name
        for m in re.finditer(r"(?<![\w])" + re.escape(name) + r"(?![\w])", text):
            spans["PERSON"].append((m.start(), m.end()))

    for m in NRIC.finditer(text):
        spans["NRIC"].append((m.start(), m.end()))
    nric_ranges = list(spans["NRIC"])

    for m in PHONE_CANDIDATE.finditer(text):
        s, e = m.start(), m.end()
        if any(s < ne and e > ns for ns, ne in nric_ranges):
            continue                             # that digit run is a MyKad, not a phone
        if "/" in text[max(0, s - 1):e + 1]:
            continue                             # dates and reference numbers
        is_org = bool(ORG_PHONE_CONTEXT.search(text[max(0, s - 22):s]))
        if not is_org or org_phones:
            spans["PHONE"].append((s, e))

    spans["ADDRESS"] = char_spans(text, offsets, ADDRESS_CONTEXTS)
    if org_addresses:
        for m in ORG_ADDRESS.finditer(text):
            spans["ADDRESS"].append((m.start(1), m.end(1)))

    return label_tokens(offsets, spans), spans, text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf")
    parser.add_argument("--out", default="data/processed/real_benchmark_50_gold.txt")
    parser.add_argument("--org-phones", action="store_true", help="label organisation phone numbers as PHONE")
    parser.add_argument("--org-addresses", action="store_true", help="label letterhead addresses as ADDRESS")
    args = parser.parse_args()

    from pdf_ingestion import PDFIngestor
    try:
        document = PDFIngestor().extract(args.pdf)
    except Exception as e:                       # unreadable or missing PDF
        print(f"Cannot ingest {args.pdf}: {e}")
        sys.exit(1)

    totals = {"PERSON": 0, "NRIC": 0, "PHONE": 0, "ADDRESS": 0}
    problems = []
    lines = []
    for page_no, page in enumerate(document.pages, start=1):
        tokens = [t.text for t in page.tokens]
        tags, spans, _text = gold_for_page(tokens, args.org_phones, args.org_addresses)
        counts = {k: sum(1 for t in tags if t == "B-" + k) for k in totals}
        for k in totals:
            totals[k] += counts[k]
        if counts["PERSON"] == 0 or counts["ADDRESS"] == 0 or counts["NRIC"] == 0:
            problems.append((page_no, counts))
        lines.extend(f"{tok} {tag}" for tok, tag in zip(tokens, tags))
        lines.append("")                         # blank line between pages

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"pages: {len(document.pages)} | gold spans: {totals}")
    print(f"pages missing a PERSON, ADDRESS or NRIC span: {problems or 'none'}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
