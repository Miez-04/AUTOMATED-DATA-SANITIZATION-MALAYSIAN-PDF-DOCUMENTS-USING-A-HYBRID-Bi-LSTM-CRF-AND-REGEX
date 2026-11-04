"""
pdf-sanitizer/data/build_v27_corpus.py

Second layer on top of the v26 corpus. Reason, measured on the v29 model's
IOB2 dump of a 64,619-token Ministry circular (a long document with about
100 real committee names and one real address):

  168 PERSON spans
    30  contain a list marker "b." / "B." / "b" (no name connector at all):
        "Objektif b. Panduan", "KPM. b. Menetapkan", "murid. b. Panduan"
    45  names fused with agency codes, state names or titles that sit in the
        next table column: "BPSH AZLINA BINTI AHMAD BPSH JUANITA",
        "JOHOR SHAHRIZAN BIN ABDUL RAHMAN", "PAHANG PAHANG PAHANG"
    11  headings and short phrases ("TMK (ICT)", "Menghadapi Banjir")
    82  clean names (the model reads "X BIN Y" well)
  7 ADDRESS spans, 6 of them false: state-name columns ("PULAU PINANG",
  "PERLIS PERLIS FADZIL") and a contents line ("378 SENARAI RAJAH ... 381").

Cause of the first group, counted in the v25 corpus (deduplicated):
    "B." is I-PERSON 202 times and O 2 times; "b." is I-PERSON 122 times and
    O 0 times; the list markers a. b. c. appear as O only 13 times. The model
    learned "capitalised word, b., capitalised word = name" with no counter-example.

This builder therefore
  1. REMOVES the "b." / "B." name connector from the whole corpus: inside
     PERSON spans it becomes "bin" / "Bin" / "BIN" (case follows the
     neighbouring name token). "bt", "bt.", "bte", "bte.", "a/l", "a/p", "anak",
     "ak", "@" are kept: they do not collide with list markers.
  2. ADDS a layer (6 families):
       enumerations   a. b. c. / A. B. C. / i. ii. / 1. 2. / (a) (b) / 1) 2)       (all O)
       committee rows [STATE] NAME [AGENCY CODE] with titles and post-nominals       (NAME only)
       state columns  repeated state names, W.P. names, agency names                 (all O)
       front matter   contents lines with LIST OF ... and roman page numbers         (all O)
       procedure titles and bracketed abbreviations                                  (all O)
       choice lines   "A. Ada B. Tiada" style option rows                            (all O)

Independence: no sentence, name or number is copied from a test document; the
pools are generic. Torch-free.

    python -m data.build_v27_corpus ^
        --v26 data/processed/semi_synthetic_v26_ns3k.txt ^
        --out data/processed/semi_synthetic_v27_ns3k.txt

then (as before):
    python -m eval.dedupe_blocks <v27 file> <v27 dedup file>
    python -m eval.split_holdout --data <v27 dedup file> --test-ratio 0.15 ^
        --train-pool-out <train pool> --test-holdout-out <test holdout>
"""

from __future__ import annotations

import argparse
import random
from collections import Counter

from data.build_v18_corpus import Sentence, entity_token_counts, parse_iob2_file, write_iob2
from data.build_v23_corpus import _cat
from data.build_v26_corpus import (
    HEAD_WORDS,
    HEADING_STANDALONE,
    JOBS,
    STATES_UP,
    _ent,
    _text,
    _valid,
    _wrap,
    dedupe,
    layer_person,
    zero_entity_share,
)


# ---------------------------------------------------------------------
# 1. Remove the "b." / "B." connector
# ---------------------------------------------------------------------

def normalise_bin_connector(sent: Sentence) -> tuple[Sentence, int]:
    """Input : one Sentence. Output: (Sentence, number of tokens replaced).
    Inside a PERSON span "b." / "B." / "b" / "B" becomes "bin" / "Bin" / "BIN".
    Case rule: neighbouring name token ALL CAPS -> BIN; connector lower case -> bin; else Bin."""
    toks = list(sent.tokens)
    changed = 0
    for i, (tok, lab) in enumerate(zip(sent.tokens, sent.labels)):
        if lab == "I-PERSON" and tok.lower() in ("b", "b."):
            nxt = toks[i + 1] if i + 1 < len(toks) and sent.labels[i + 1] == "I-PERSON" else toks[i - 1]
            if nxt.strip(",.").isupper():
                toks[i] = "BIN"
            elif tok.islower():
                toks[i] = "bin"
            else:
                toks[i] = "Bin"
            changed += 1
    return Sentence(tokens=toks, labels=list(sent.labels)), changed


# ---------------------------------------------------------------------
# 2. Word banks (generic)
# ---------------------------------------------------------------------
VERBS = ["Merancang", "Melaksanakan", "Memantau", "Melaporkan", "Mengenal pasti", "Menyediakan", "Menyemak",
         "Mengesahkan", "Merujuk", "Menghubungi", "Mengemas kini", "Mendokumentasikan", "Menyelaras", "Menilai"]
OBJECTS = ["program sekolah", "rekod kehadiran", "laporan bulanan", "pelan tindakan", "borang kebenaran",
           "senarai semak", "prosedur kecemasan", "jadual tugas", "bahan sokongan", "maklumat murid",
           "kemudahan asrama", "peralatan makmal", "garis panduan", "minit mesyuarat"]
TITLE_PAIRS = ["Matlamat Khusus", "Garis Tindakan", "Skop", "Pelaksanaan", "Tanggungjawab", "Langkah Awal", "Rujukan",
               "Kriteria", "Tindakan Susulan", "Maklum Balas", "Impak", "Lampiran"]
ABBR = ["JPN", "PPD", "KPM", "SOP", "PIBG", "JKR", "LPM", "KSSR", "PAJSK", "BPSH", "MOE", "PTM"]
MARKER_SETS = [["a.", "b.", "c.", "d.", "e."], ["A.", "B.", "C.", "D."], ["i.", "ii.", "iii.", "iv.", "v."],
               ["1.", "2.", "3.", "4."], ["(a)", "(b)", "(c)", "(d)"], ["1)", "2)", "3)", "4)"],
               ["(i)", "(ii)", "(iii)"], ["•", "•", "•", "•"]]
AGENCY_CODES = ["BPK", "BPSH", "BSKK", "BPPDP", "BPL", "BPI", "BPM", "PUU", "UPK", "SPP", "BKP", "BPA", "BTPN", "BPG", "SPPB"]
PERSON_PREFIX = ["DR.", "DATO’", "DATIN", "HAJI", "HJH", "PROF.", "TUAN", "PUAN", "ENCIK", "Dr.", "Dato’", "Prof."]
POST_NOMINALS = ["K.M.N.,", "B.S.K.", "A.M.N.", "P.P.T.", "A.B.S.", "D.P.M.P.", "J.S.M."]
ROLE_UP = ["PENASIHAT", "PENGERUSI", "NAIB PENGERUSI", "PENYELARAS", "SETIAUSAHA", "AHLI", "EDITOR", "PENOLONG PENGARAH"]
POSITION_UP = ["KETUA PENGARAH PENDIDIKAN", "TIMBALAN PENGARAH", "PENOLONG PENGARAH", "KETUA PENOLONG PENGARAH",
               "PENGARAH BAHAGIAN", "PEGAWAI PERKHIDMATAN PENDIDIKAN", "KETUA UNIT"]
AGENCY_NAMES = ["JABATAN PENDIDIKAN NEGERI", "PEJABAT PENDIDIKAN DAERAH", "KEMENTERIAN KESIHATAN MALAYSIA",
                "JABATAN KEBAJIKAN MASYARAKAT", "POLIS DIRAJA MALAYSIA", "JABATAN PENGANGKUTAN JALAN",
                "BAHAGIAN SUMBER MANUSIA", "INSTITUT LATIHAN", "MAJLIS PERBANDARAN"]
WP_NAMES = ["W.P. KUALA LUMPUR", "W.P. LABUAN", "W.P. PUTRAJAYA"]
FRONT_ITEMS = ["SENARAI JADUAL", "SENARAI LAMPIRAN", "SENARAI SINGKATAN", "INDEKS", "PENGHARGAAN", "RUJUKAN", "BIBLIOGRAFI",
               "SENARAI GAMBAR", "LATAR BELAKANG", "ISI KANDUNGAN"]
PROCEDURE_VERBS = ["Mengurus", "Melaporkan", "Menangani", "Memantau", "Menyelia", "Mengendalikan"]
PROCEDURE_NOUNS = ["Kehilangan Aset", "Aduan Awam", "Gangguan Bekalan", "Kemalangan Kenderaan", "Kerosakan Bangunan",
                   "Tunjuk Perasaan", "Pencemaran Air", "Kecurian"]
ABBR_PAIRS = [("KPI", "Key Performance Indicator"), ("HEM", "Hal Ehwal Murid"), ("PPKI", "Program Pendidikan Khas Integrasi"),
              ("PBD", "Pentaksiran Bilik Darjah"), ("SPM", "Sijil Pelajaran Malaysia"), ("LDP", "Lawatan Dalam Perkhidmatan"),
              ("RPH", "Rancangan Pengajaran Harian"), ("KOKO", "Kokurikulum")]
CHOICES = [("Ada", "Tiada"), ("Lulus", "Gagal"), ("Ya", "Tidak"), ("Sesuai", "Tidak Sesuai"), ("Lengkap", "Tidak Lengkap")]


def _item_text(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.40:
        return rng.choice(TITLE_PAIRS)
    if r < 0.85:
        tail = f" {rng.choice(OBJECTS)}"
        if rng.random() < 0.4:
            tail += f" {rng.choice(['oleh', 'bersama', 'kepada'])} {rng.choice(ABBR)}"
        return rng.choice(VERBS) + tail
    return rng.choice(ABBR) + " " + rng.choice(OBJECTS)


# ---------------------------------------------------------------------
# 3. Families
# ---------------------------------------------------------------------

def enumeration(rng: random.Random) -> Sentence:
    """a. ... b. ... c. ...: items end with '.' or ';' or nothing, so a marker can follow a full stop."""
    markers = rng.choice(MARKER_SETS)
    n = rng.randint(2, min(5, len(markers)))
    sep = rng.choice([".", ".", ";", ""])
    parts = []
    for k in range(n):
        item = _item_text(rng)
        if rng.random() < 0.15:
            item = item.upper()
        parts.append(f"{markers[k]} {item}{sep}")
    text = " ".join(parts)
    if rng.random() < 0.5:  # an intro phrase before the list
        text = f"{rng.choice(['Peranan', 'Tugas', 'Langkah', 'Syarat', 'Kriteria'])} {rng.choice(OBJECTS)} : " + text
    if rng.random() < 0.3:  # a short heading line after the list, as in flattened PDF pages
        text += " " + " ".join(rng.sample(HEAD_WORDS, 2))
    return _text(text)


def choice_row(rng: random.Random) -> Sentence:
    """'A. Ada B. Tiada' option rows and 'Ada/Tiada B.' style fragments."""
    yes, no = rng.choice(CHOICES)
    parts = [f"{rng.choice(['Status', 'Semakan', 'Keputusan'])} : {yes}/{no}"]
    letters = rng.choice(["AB", "ABC", "ABCD"])
    for k, letter in enumerate(letters):
        parts.append(f"{letter}. {_item_text(rng)}")
    if rng.random() < 0.5:
        parts.insert(rng.randint(1, len(parts)), rng.choice(['B.', 'A.', 'C.']) + " " + rng.choice(TITLE_PAIRS))
    return _text(" ".join(parts))


def _name_entry(rng: random.Random) -> Sentence:
    """One table entry: optional leading state or agency code, NAME, optional trailing code / post-nominal."""
    toks = layer_person(rng)
    if rng.random() < 0.55:
        toks = [t.upper() for t in toks]
    lead = rng.random()
    trail = rng.random()
    parts = []
    if lead < 0.25:
        parts.append(_text(rng.choice(STATES_UP + WP_NAMES)))
    elif lead < 0.50:
        parts.append(_text(rng.choice(AGENCY_CODES)))
    elif lead < 0.60:
        parts.append(_text(rng.choice(PERSON_PREFIX)))
    parts.append(_ent("PERSON", toks))
    if trail < 0.40:
        parts.append(_text(rng.choice(AGENCY_CODES)))
    elif trail < 0.55:
        parts.append(_text(rng.choice(POST_NOMINALS)))
    elif trail < 0.70:
        parts.append(_text(rng.choice(STATES_UP)))
    elif trail < 0.92:
        parts.append(_text(rng.choice(POSITION_UP)))   # two names almost never touch: a column cell sits between them
    return _cat(parts)


def committee_rows(rng: random.Random) -> Sentence:
    parts = []
    if rng.random() < 0.7:
        parts.append(_text(rng.choice(["AHLI JAWATANKUASA PENGGUBAL", "JAWATANKUASA PELAKSANA", "SENARAI PENYELARAS", "PANEL PENULIS"])))
    for _ in range(rng.randint(2, 6)):
        if rng.random() < 0.30:
            parts.append(_text(rng.choice(ROLE_UP)))
        parts.append(_name_entry(rng))
        if rng.random() < 0.35:
            parts.append(_text(rng.choice(POSITION_UP)))
        if rng.random() < 0.15:
            parts.append(_text(rng.choice(AGENCY_NAMES)))
    return _cat(parts)


def state_column(rng: random.Random) -> Sentence:
    items = []
    for _ in range(rng.randint(4, 12)):
        name = rng.choice(STATES_UP + WP_NAMES)
        items.extend([name] * rng.choice([1, 1, 2, 3]))
    if rng.random() < 0.4:
        items.insert(rng.randint(0, len(items)), rng.choice(AGENCY_NAMES))
    return _text(" ".join(items))


def front_matter(rng: random.Random) -> Sentence:
    roman = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"]
    parts, page = [], rng.randint(1, 8)
    for _ in range(rng.randint(3, 7)):
        name = rng.choice(FRONT_ITEMS + HEADING_STANDALONE)
        parts.append(f"{name} {roman[page % len(roman)] if rng.random() < 0.3 else rng.randint(page, page + 40)}")
        page += rng.randint(1, 30)
    return _text(" ".join(parts))


def procedure_title(rng: random.Random) -> Sentence:
    r = rng.random()
    if r < 0.45:
        text = f"Carta Alir Prosedur {rng.choice(PROCEDURE_VERBS)} {rng.choice(PROCEDURE_NOUNS)}"
        if rng.random() < 0.4:
            text += f" {rng.randint(100, 400)}"
    elif r < 0.80:
        abbr, full = rng.choice(ABBR_PAIRS)
        text = f"{rng.choice(['Guru Penyelaras', 'Penyelaras', 'Pegawai', 'Ketua Unit'])} {abbr} ({full})"
    else:
        text = " ".join(rng.sample(JOBS, 2)) + f" {rng.choice(ABBR)} ({rng.choice(['Pegawai', 'Ketua', 'Penyelia'])})"
    return _text(text)


FAMILIES = [(enumeration, 0.30), (choice_row, 0.05), (committee_rows, 0.30), (state_column, 0.08),
            (front_matter, 0.08), (procedure_title, 0.19)]


def build_layer(rng: random.Random, count: int) -> list[Sentence]:
    funcs = [f for f, _ in FAMILIES]
    weights = [w for _, w in FAMILIES]
    out = []
    for _ in range(count):
        core = rng.choices(funcs, weights)[0](rng)
        out.append(_wrap(rng, core, 0.15) if core.labels.count("O") != len(core.labels) else core)
    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v26", default="data/processed/semi_synthetic_v26_ns3k.txt")
    ap.add_argument("--layer-count", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=27)
    ap.add_argument("--keep-b-connector", action="store_true", help="skip step 1 (keep b. / B. as name connectors)")
    ap.add_argument("--out", default="data/processed/semi_synthetic_v27_ns3k.txt")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    raw = parse_iob2_file(args.v26)

    replaced, base = 0, []
    for s in raw:
        if args.keep_b_connector:
            base.append(s)
        else:
            s2, n = normalise_bin_connector(s)
            replaced += n
            base.append(s2)
    base = dedupe(base)

    layer = build_layer(rng, args.layer_count)
    if not args.keep_b_connector:
        layer = [normalise_bin_connector(s)[0] for s in layer]
    layer = dedupe(layer)
    bad = [s for s in layer if not _valid(s)]
    assert not bad, f"{len(bad)} invalid layer blocks, first: {bad[0].tokens[:20]} {bad[0].labels[:20]}"
    combined = base + layer
    rng.shuffle(combined)

    still_b = sum(1 for s in combined for t, l in zip(s.tokens, s.labels) if l == "I-PERSON" and t.lower() in ("b", "b."))
    print(f"v26 input: {len(raw)} | connector tokens rewritten (b./B. -> bin/Bin/BIN): {replaced} | "
          f"after dedupe: {len(base)} | new layer: {len(layer)} | total: {len(combined)}")
    print(f"'b.'/'B.' still inside PERSON spans: {still_b}")
    print(f"zero-entity share: base {zero_entity_share(base):.1%} -> combined {zero_entity_share(combined):.1%} "
          f"(layer alone {zero_entity_share(layer):.1%})")
    before, after = entity_token_counts(base), entity_token_counts(combined)
    for tag in sorted(set(before) | set(after)):
        print(f"  {tag:8s} {before.get(tag, 0):7d} -> {after.get(tag, 0):7d}")
    write_iob2(combined, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
