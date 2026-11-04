"""
pdf-sanitizer/data/build_v26_corpus.py

Adds a FORMS / TABLES / MALAY-NOTICE layer on top of the (deduplicated) v25
corpus. Reason, measured on two real documents with the v27 and v28 models:

  * PBS guideline PDF (a document with ~12 names and NO postal address):
        v27: 19 ADDRESS + 6 PERSON false-positive spans
        v28: 45 ADDRESS + 9 PERSON false-positive spans
    The model called these names / addresses: "Kelas : TAHUN ENAM",
    "NAMA KELAS : <class name>", "Pilihan Kerjaya : Guru, Kaunselor,", all-caps
    headings, contents lines with page-number runs, mark / grade / module-code
    tables and rubric rows containing state words ("PERINGKAT NEGERI").
  * Mixed Malay / English sample pack: 6 of 12 name mentions missed, all in
    plain Malay notices ("iaitu <name>,", "atas nama <name>,", "pemilik
    berdaftar iaitu <name>,"), a name that starts a line under a section
    heading, and names whose tokens never appear as names in the corpus.
    In v25, "iaitu", "pemilik", "pemohon" and "pengadu" were NEVER the word
    before a PERSON span.

Layer content (label O unless an entity is named):
  Negatives (about 58 % of the new blocks have no entity at all)
    1. contents lines with page-number runs
    2. rubric tables (marks in brackets, level words, state names)
    3. module-code / grade-band / mark tables
    4. all-caps abbreviation header rows
    5. occupation lists and Title-Case phrases after a label
    6. all-caps headings
  Positives
    7. form blocks "Label : value": Nama = PERSON, Nama Kelas / Kelas / Kod
       Sekolah / Pilihan Kerjaya / Umur ... = O (same slot, different label)
    8. Malay and code-mixed notices with the name contexts listed above
    9. rural / longhouse / P.O. Box addresses (ADDRESS)
   10. a name placed between a heading and a job title (title words stay O)
   11. numbered student-list rows (number, NAME, NRIC, school code)
  Extra name pool: Malay, Indian, Chinese and Iban/Sarawak names, upper / title
  case, connector variants (bin / Bin / BIN, a/l, A/P, anak, ak).

Independence: no sentence, name or number is copied from a test document. The
pools are generic. HELD_OUT_NAMES below lists the full names that appear in
the evaluation documents; generated names equal to one of them are rejected.
Imports nothing from regex_engine. Torch-free.

    python -m data.build_v26_corpus ^
        --v25 data/processed/semi_synthetic_v25_ns3k.txt ^
        --out data/processed/semi_synthetic_v26_ns3k.txt

then, as before:
    python -m eval.dedupe_blocks <v26 file> <v26 dedup file>
    python -m eval.split_holdout --data <v26 dedup file> --test-ratio 0.15 ^
        --train-pool-out <train pool> --test-holdout-out <test holdout>
"""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter

from data.build_v18_corpus import Sentence, entity_token_counts, parse_iob2_file, write_iob2
from data.build_v23_corpus import _cat, _entity, _o, address_tokens, nric_tokens, person_tokens, phone_tokens
from data.build_v24_corpus import mixed_person_tokens

# Full names (lower case, no trailing punctuation) that occur in the evaluation documents.
HELD_OUT_NAMES = {
    "muhammad alif bin rahmat", "wong kar wai", "logeswaran a/l subramaniam", "lim mei yee",
    "nurul ain binti zakaria", "muhammad farhan bin zulkifli", "wan ahmad akmal bin wan said",
    "besi anak lelaki jikun", "siti nor aisyah binti abdullah", "tan mei ling", "rusmini binti ku ahmad",
    "haji roslan bin abu hassan", "dilya essayue athira binti hassan", "alif bin aziz", "aiman bin taro",
    "ahmad shah jibriel bin mohd norhalini", "ahmad aryan al kahfi bin ahmad hazli", "nurul farah binti mohd",
    "muhammad aidiel bin hisham", "aisyah husna binti salim", "ain nur aisyah binti kasim",
}

# ---------------------------------------------------------------------
# Extra name pool
# ---------------------------------------------------------------------
_MALAY_M = ["Hafizuddin", "Syazwan", "Khairul", "Arif", "Zaidi", "Hakimi", "Irfan", "Danial", "Faiz", "Amirul",
            "Shahrul", "Nazmi", "Taufik", "Fikri", "Rizal", "Azrul", "Hanif", "Luqman", "Badrul", "Ikhwan"]
_MALAY_F = ["Nadhirah", "Syafiqah", "Aina", "Farhana", "Hidayah", "Najwa", "Izzati", "Sofea", "Balqis", "Atiqah",
            "Husna", "Liyana", "Mawaddah", "Qistina", "Sakinah", "Umairah", "Wardah", "Yasmin", "Zulaikha", "Aliya"]
_MALAY_FATHER = ["Salleh", "Taha", "Idris", "Hamid", "Yaakob", "Bakar", "Daud", "Ismail", "Othman", "Jaafar",
                 "Mansor", "Osman", "Harun", "Yusof", "Ghani", "Kassim", "Latif", "Samad", "Ramli", "Zainal",
                 "Mahmud", "Hashim", "Rashid", "Awang", "Deraman"]
_INDIAN_M = ["Thanabalan", "Murugesan", "Gunasegaran", "Selvarajah", "Kumaravel", "Rajendran", "Sivakumar",
             "Vijayan", "Prakash", "Ganesan", "Nadarajah", "Balakrishnan", "Devadas", "Muniandy", "Paramasivam"]
_INDIAN_F = ["Kalaivani", "Revathi", "Saraswathy", "Thilagavathy", "Punitha", "Mahalakshmi", "Jayanthi",
             "Sumathi", "Vasantha", "Santhi"]
_INDIAN_FATHER = ["Arumugam", "Ramasamy", "Veerasamy", "Krishnan", "Maniam", "Pillay", "Chelliah", "Kandasamy",
                  "Sinnathamby", "Perumal", "Nair", "Subbiah"]
_CHINESE_SURNAME = ["Tan", "Lim", "Lee", "Ong", "Chong", "Yap", "Goh", "Teoh", "Chew", "Phang", "Khoo", "Loh",
                    "Ho", "Wee", "Chin", "Foo"]
_CHINESE_GIVEN = ["Wei", "Jie", "Hui", "Ling", "Mei", "Yee", "Siew", "Kah", "Chee", "Boon", "Kok", "Yen", "Zhi",
                  "Xin", "Jia", "Hao", "Ee", "Sze", "Pei", "Kai", "Shan", "Ying", "Jun", "Ken"]
_IBAN_GIVEN = ["Bujang", "Lugu", "Ngelai", "Entri", "Sandai", "Unyang", "Juna", "Dunstan", "Jonathan", "Rita",
               "Linda", "Gerald", "Nyaring", "Mikai", "Belinda", "Dayang", "Awang", "Jenny"]
_IBAN_FATHER = ["Jawa", "Sibat", "Nanta", "Lunsa", "Ripai", "Embol", "Mapai", "Tugang", "Kanyan", "Jubang"]


def _blocked(tokens: list[str]) -> bool:
    text = " ".join(t.strip(",.;:") for t in tokens).lower()
    return text in HELD_OUT_NAMES


def extra_person_tokens(rng: random.Random) -> list[str]:
    """One generated name, 2 to 5 tokens, from the layer's own pools."""
    style = rng.choice(["malay_m", "malay_m", "malay_f", "malay_f", "indian_m", "indian_f", "chinese", "iban"])
    if style == "malay_m":
        toks = [rng.choice(_MALAY_M)] + ([rng.choice(_MALAY_M)] if rng.random() < 0.4 else []) + \
               ["Bin", rng.choice(_MALAY_FATHER)] + ([rng.choice(_MALAY_FATHER)] if rng.random() < 0.2 else [])
    elif style == "malay_f":
        toks = [rng.choice(_MALAY_F)] + ([rng.choice(_MALAY_F)] if rng.random() < 0.4 else []) + \
               ["Binti", rng.choice(_MALAY_FATHER)] + ([rng.choice(_MALAY_FATHER)] if rng.random() < 0.2 else [])
    elif style == "indian_m":
        toks = [rng.choice(_INDIAN_M), rng.choice(["A/L", "a/l", "s/o"]), rng.choice(_INDIAN_FATHER)]
    elif style == "indian_f":
        toks = [rng.choice(_INDIAN_F), rng.choice(["A/P", "a/p", "d/o"]), rng.choice(_INDIAN_FATHER)]
    elif style == "chinese":
        toks = [rng.choice(_CHINESE_SURNAME)] + [rng.choice(_CHINESE_GIVEN) for _ in range(rng.choice([1, 2, 2]))]
        if rng.random() < 0.15:
            toks = toks[1:] + [toks[0]]  # given names first, surname last
    else:
        toks = [rng.choice(_IBAN_GIVEN), rng.choice(["anak", "ak", "Anak"]), rng.choice(_IBAN_FATHER)]
    r = rng.random()
    if r < 0.25:
        toks = [t.upper() for t in toks]
    elif r < 0.40:
        toks = [t.lower() if t.lower() in {"bin", "binti", "a/l", "a/p", "anak"} else t for t in toks]
    return toks


def layer_person(rng: random.Random) -> list[str]:
    for _ in range(30):
        toks = extra_person_tokens(rng) if rng.random() < 0.55 else \
            (mixed_person_tokens(rng, 0.3) if rng.random() < 0.7 else person_tokens(rng))
        if not _blocked(toks):
            return toks
    return extra_person_tokens(rng)


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def _text(s: str) -> Sentence:
    t, l = _o(s)
    return Sentence(tokens=t, labels=l)


def _ent(kind: str, tokens: list[str], trailing: str = "") -> Sentence:
    t, l = _entity(kind, tokens, trailing)
    return Sentence(tokens=t, labels=l)


def _trail(tokens: list[str], ch: str) -> str:
    """Trailing punctuation to attach: only when the last token has none yet."""
    if ch and tokens and tokens[-1][-1] not in ",.;:":
        return ch
    return ""


def _up(s: str) -> str:
    return s.upper()


# ---------------------------------------------------------------------
# Word banks (generic)
# ---------------------------------------------------------------------
STATES_UP = ["JOHOR", "KEDAH", "KELANTAN", "MELAKA", "PAHANG", "PERAK", "PERLIS", "PULAU PINANG", "SABAH",
             "SARAWAK", "SELANGOR", "TERENGGANU", "NEGERI SEMBILAN", "WILAYAH PERSEKUTUAN"]
HEAD_WORDS = ["DASAR", "PELAKSANAAN", "PENGURUSAN", "PENTADBIRAN", "KURIKULUM", "PENILAIAN", "LATIHAN", "PEMANTAUAN",
              "KUALITI", "STRATEGI", "PERANCANGAN", "SUMBER", "PEMBANGUNAN", "PROFESIONAL", "KOMUNIKASI",
              "TEKNOLOGI", "INOVASI", "KEPIMPINAN", "KOKURIKULUM", "PELAPORAN", "PENGESAHAN", "RUJUKAN"]
HEADING_STANDALONE = ["OBJEKTIF", "PRINSIP", "MATLAMAT", "SKOP", "DEFINISI", "RUMUSAN", "PENDAHULUAN",
                      "LATAR BELAKANG", "KANDUNGAN", "SENARAI SEMAK", "LAMPIRAN", "GLOSARI", "SINGKATAN",
                      "FUNGSI UTAMA", "KRITERIA PENILAIAN", "CIRI PENTING", "RINGKASAN EKSEKUTIF", "PENUTUP"]
ROLE_WORDS = ["PENGERUSI", "NAIB PENGERUSI", "SETIAUSAHA", "BENDAHARI", "AHLI JAWATANKUASA", "AHLI BIASA",
              "JURULATIH", "PENASIHAT", "PENGURUS", "PENOLONG SETIAUSAHA"]
LEVEL_WORDS = ["PERINGKAT SEKOLAH", "PERINGKAT DAERAH", "PERINGKAT NEGERI", "PERINGKAT KEBANGSAAN", "ANTARABANGSA",
               "PERINGKAT ZON", "PERINGKAT BAHAGIAN", "PERINGKAT DAERAH/ZON"]
RESULT_WORDS = ["JOHAN", "NAIB JOHAN", "KETIGA", "KEEMPAT", "PENYERTAAN", "KEHADIRAN", "KOMITMEN", "PRESTASI"]
BAND_WORDS = ["Cemerlang", "Kepujian", "Baik", "Memuaskan", "Lemah", "Minimum", "Sangat Baik", "Terbaik"]
SKILLS = ["Pendawaian", "Kimpalan", "Pembuatan", "Tanaman", "Katering", "Jahitan", "Elektronik", "Penyejukan",
          "Automotif", "Seni Reka", "Multimedia", "Pembinaan", "Perhotelan", "Akuakultur"]
SUBJECTS = ["Matematik", "Sains", "Bahasa Inggeris", "Sejarah", "Geografi", "Pendidikan Seni", "Reka Cipta",
            "Pendidikan Jasmani", "Ekonomi", "Perakaunan", "Biologi", "Kimia", "Fizik", "Pendidikan Moral"]
JOBS = ["Guru", "Jurutera", "Akauntan", "Doktor", "Jururawat", "Peguam", "Arkitek", "Juruteknik", "Pegawai Tadbir",
        "Pengurus Projek", "Pereka Grafik", "Penulis", "Pensyarah", "Penyelidik", "Pegawai Polis", "Juruaudit",
        "Pustakawan", "Chef", "Juruterbang", "Usahawan", "Pegawai Kewangan", "Penterjemah", "Kaunselor",
        "Pengaturcara", "Pegawai Pemasaran", "Jurukur", "Penolong Pegawai Perubatan"]
PHRASE_A = ["Markah", "Skor", "Tahap", "Pencapaian", "Kekerapan", "Purata", "Julat", "Kategori", "Bilangan", "Peratus"]
PHRASE_B = ["Tertinggi", "Terendah", "Keseluruhan", "Utama", "Sederhana", "Semasa", "Terkini", "Minimum", "Maksimum"]
CLASS_ADJ = ["ARIF", "BIJAK", "CEKAP", "DINAMIK", "HARMONI", "KREATIF", "MURNI", "TEKUN", "UNGGUL", "WIRA", "AMANAH",
             "BESTARI", "CERIA", "GIGIH", "INTELEK", "MAWAR", "SEROJA", "TELADAN"]
NUM_WORDS = ["SATU", "DUA", "TIGA", "EMPAT", "LIMA", "ENAM"]
TITLES_OFFICIAL = ["Ketua Pengarah Pendidikan", "Pengarah Bahagian Perancangan", "Timbalan Setiausaha Bahagian",
                   "Ketua Jabatan Operasi", "Pengerusi Lembaga Pengarah", "Pegawai Daerah", "Naib Canselor",
                   "Pengetua Kolej", "Pengarah Eksekutif", "Setiausaha Kerja"]
HEADINGS_PERSONAL = ["KATA PENGANTAR", "PERUTUSAN PENGARAH", "UCAPAN PENGERUSI", "MESSAGE FROM THE DIRECTOR",
                     "FOREWORD", "SAMBUTAN KETUA", "KATA-KATA ALUAN"]
SCHOOL_PLACE = ["Taman Harmoni", "Bandar Baru", "Seri Murni", "Kampung Tengah", "Bukit Indah", "Jaya Utama", "Pasir Puteh"]


def _school_code(rng: random.Random) -> str:
    return "".join(rng.choice("ABCDEFGHJKLMNPRSTVW") for _ in range(3)) + str(rng.randint(1000, 9999))


def _class_text(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.5:
        return f"{rng.randint(1, 6)} {rng.choice(CLASS_ADJ)}"
    if r < 0.8:
        return f"TINGKATAN {rng.randint(1, 5)} {rng.choice(CLASS_ADJ)}"
    return f"TAHUN {rng.choice(NUM_WORDS)} {rng.choice(CLASS_ADJ)}"


def _title_phrase(rng: random.Random) -> str:
    return f"{rng.choice(PHRASE_A)} {rng.choice(PHRASE_B)}"


# ---------------------------------------------------------------------
# Negative families
# ---------------------------------------------------------------------

def contents_lines(rng: random.Random) -> Sentence:
    parts: list[str] = []
    page = rng.randint(1, 20)
    for chap in range(1, rng.randint(3, 6)):
        parts.append(f"BAB {chap} {' '.join(rng.sample(HEAD_WORDS, rng.randint(1, 3)))} {page}")
        page += rng.randint(2, 15)
        for sub in range(1, rng.randint(2, 5)):
            parts.append(f"{chap}.{sub} {rng.choice(PHRASE_A)} {rng.choice(PHRASE_B).lower()} {page}")
            page += rng.randint(0, 4)
    if rng.random() < 0.4:  # a bare run of page numbers, as a flattened index column
        parts.append(" ".join(str(rng.randint(1, 90)) for _ in range(rng.randint(5, 12))))
    return _text(" ".join(parts))


def rubric_table(rng: random.Random) -> Sentence:
    cells: list[str] = []
    for _ in range(rng.randint(3, 6)):
        kind = rng.random()
        if kind < 0.35:
            cells.append(f"({rng.randint(2, 20)}) {rng.choice(ROLE_WORDS)}")
        elif kind < 0.70:
            cells.append(f"{rng.choice(RESULT_WORDS)} [ {rng.choice(STATES_UP + ['DAERAH', 'BAHAGIAN', 'KAMPUS'])} ] ({rng.randint(4, 20)})")
        else:
            cells.append(f"{rng.choice(LEVEL_WORDS)} ({rng.randint(4, 20)})")
    tail = " ".join(f"{rng.randint(4, 40)} ({rng.randint(10, 60)})" for _ in range(rng.randint(1, 4)))
    return _text(" ".join(cells) + " " + tail)


def code_table(rng: random.Random) -> Sentence:
    kind = rng.random()
    if kind < 0.40:  # module / paper code list
        rows = []
        for _ in range(rng.randint(4, 9)):
            code = f"{rng.randint(1000, 9999)}/{rng.randint(1, 3)}"
            rows.append(f"{rng.randint(1, 80)} {code} {rng.randint(2, 4)} Modul {rng.choice(SKILLS)}")
        return _text(" ".join(rows))
    if kind < 0.70:  # grade bands: start, letters and wording vary
        top, rows = rng.choice([100, 100, 95, 90, 99]), []
        letters = rng.choice(["ABCDEF", "ABCDE", "ABCD", "ABCDEG"])
        words = rng.sample(BAND_WORDS, len(letters))
        for letter, word in zip(letters, words):
            low = max(0, top - rng.randint(8, 20))
            rows.append(f"{low + 1} - {top} " + (f"{letter} {word}" if rng.random() < 0.5 else f"{word} {letter}"))
            top = low
        return _text(" ".join(rows))
    rows = []  # subject mark table
    for _ in range(rng.randint(4, 8)):
        rows.append(f"{rng.randint(1, 60)} {rng.choice(SUBJECTS)} {rng.randint(0, 100)} {rng.choice('ABCDE')} "
                    f"{rng.randint(1, 5)} {rng.randint(10, 99)}/{rng.randint(10, 99)}")
    return _text(" ".join(rows))


def abbreviation_header(rng: random.Random) -> Sentence:
    codes = ["".join(rng.choice("ABCDEFGHIKLMNPRSTVXZ") for _ in range(rng.choice([2, 3, 3, 4]))) for _ in range(rng.randint(6, 12))]
    parts = [rng.choice(["TARIKH", "BIL", "NO.", "BULAN"]), " ".join(codes)]
    if rng.random() < 0.5:
        parts.append(f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(2022, 2026)} " + " ".join(str(rng.randint(0, 99)) for _ in range(len(codes))))
    return _text(" ".join(parts))


def occupation_list(rng: random.Random) -> Sentence:
    n = rng.randint(3, 8)
    jobs = ", ".join(rng.sample(JOBS, n))
    lead = rng.choice(["Pilihan Kerjaya :", "Kerjaya yang sesuai :", "Contoh kerjaya -", "Bidang kerjaya :", ""])
    parts = [lead, jobs + rng.choice([",", ".", ""])]
    if rng.random() < 0.6:
        parts.append(f"{_title_phrase(rng)} - {_title_phrase(rng)} {'.' * rng.randint(8, 30)} ( )")
    return _text(" ".join(p for p in parts if p))


def heading_caps(rng: random.Random) -> Sentence:
    n = rng.randint(2, 6)
    words = []
    for _ in range(n):
        words.append(rng.choice(HEADING_STANDALONE) if rng.random() < 0.5 else " ".join(rng.sample(HEAD_WORDS, rng.randint(1, 2))))
    if rng.random() < 0.4:
        words.append(f"{rng.randint(1, 60)}")
    return _text(" ".join(words))


NEGATIVES = [(contents_lines, 0.12), (rubric_table, 0.12), (code_table, 0.12), (abbreviation_header, 0.06),
             (occupation_list, 0.08), (heading_caps, 0.06)]


def _random_negative(rng: random.Random) -> Sentence:
    funcs = [f for f, _ in NEGATIVES]
    return rng.choices(funcs, [w for _, w in NEGATIVES])[0](rng)


def _wrap(rng: random.Random, core: Sentence, p: float = 0.4) -> Sentence:
    """Put a negative fragment before and/or after a positive block, so a name or
    address has to be found among headings and tables, not on its own."""
    parts = []
    if rng.random() < p:
        parts.append(_random_negative(rng))
    parts.append(core)
    if rng.random() < p / 2:
        parts.append(_random_negative(rng))
    return _cat(parts)


# ---------------------------------------------------------------------
# Positive families
# ---------------------------------------------------------------------

def layer_address(rng: random.Random) -> list[str]:
    r = rng.random()
    if r < 0.12:  # longhouse
        river = rng.choice(["Sungai", "Batang"]) + " " + rng.choice(["Rejang", "Lemanak", "Kemena", "Skrang", "Katibas", "Baram"])
        text = (f"Rumah Panjang {rng.choice(_IBAN_FATHER)}, {river}, "
                f"{rng.choice(['96000 Sibu', '97000 Bintulu', '98000 Miri', '96800 Kapit', '95000 Sri Aman', '98700 Limbang'])}, Sarawak")
    elif r < 0.20:  # village / mukim
        text = (f"Kampung {rng.choice(['Sungai Pinang', 'Pasir Mas', 'Tanjung Aru', 'Bukit Rambai', 'Telok Gong'])}, "
                f"Mukim {rng.randint(1, 12)}, {rng.choice(['21000 Kuala Terengganu', '16150 Kota Bharu', '75450 Ayer Keroh', '08000 Sungai Petani'])}, "
                f"{rng.choice(['Terengganu', 'Kelantan', 'Melaka', 'Kedah'])}")
    elif r < 0.26:  # P.O. box
        text = f"Peti Surat {rng.randint(10, 9999)}, {rng.choice(['50728 Kuala Lumpur', '80000 Johor Bahru', '40000 Shah Alam', '88999 Kota Kinabalu'])}"
    else:
        return address_tokens(rng)
    toks = text.split()
    if rng.random() < 0.25:
        toks = [t.upper() for t in toks]
    return toks


_SLOT = re.compile(r"\{([PNAT])([,.]?)\}")
_GEN = {"P": ("PERSON", layer_person), "N": ("NRIC", nric_tokens), "A": ("ADDRESS", layer_address), "T": ("PHONE", phone_tokens)}

NOTICE_TEMPLATES = [
    "Pihak kami telah menerima aduan daripada {P,} yang beralamat di {A.}",
    "Borang ini diisi oleh {P,} dengan nombor MyKad {N,} dan alamat {A.}",
    "Cukai pintu bagi rumah tersebut didaftarkan atas nama {P,} sejak tahun {Y}.",
    "Pemilik berdaftar kenderaan tersebut ialah {P,} no. pengenalan {N.}",
    "Pelanggan iaitu {P,} telah membuat bayaran penuh pada bulan lepas.",
    "Kontrak ini ditandatangani oleh {P,} mewakili pihak penyewa yang tinggal di {A.}",
    "Kami ingin memaklumkan bahawa akaun milik {P,} akan ditutup tidak lama lagi.",
    "Penama dalam polisi ini ialah {P,} {N,} hubungan : isteri.",
    "Pesakit bernama {P,} berumur {M} tahun telah dimasukkan ke wad.",
    "Pengesahan penerimaan telah dihantar kepada {P,} melalui talian {T.}",
    "Saya {P,} no. K/P {N,} ingin membuat rayuan berkaitan keputusan tersebut.",
    "Sijil ini dikeluarkan kepada {P,} bagi menyokong permohonan yang beralamat di {A.}",
    "Penama : {P} No. KP : {N} Alamat Surat : {A} Tel : {T}",
    "Kepada : {P} {A}",
    "Tuan / Puan {P,} dimaklumkan bahawa permohonan anda telah diluluskan.",
    "Waris yang berdaftar ialah {P,} hubungan anak kandung kepada si mati.",
    "Notis ini dihantar kepada penghuni rumah iaitu {P,} untuk tindakan segera.",
    "Saksi pertama {P,} dan saksi kedua {P,} telah hadir semasa prosiding.",
    "Penerima bantuan bagi bulan ini ialah {P,} {P,} dan {P.}",
    "Rekod menunjukkan bahawa pelajar bernama {P,} kelas {C} telah mendapat markah {M}.",
    "Keterangan telah diambil daripada saksi bernama {P,} {N.}",
    "Pihak tuan tanah iaitu {P,} bersetuju untuk menyewakan premis di {A.}",
    "Untuk makluman, contractor kita iaitu {P,} IC no {N,} sudah submit borang di {A.}",
    "Please contact {P} at {T.} for further clarification.",
    "The applicant, {P,} holding NRIC {N,} resides at {A.}",
    "Kindly note that the account holder {P,} has been notified.",
    "Nama pemohon yang tertera dalam borang ialah {P.}",
    "Surat ini ditujukan kepada {P,} yang bertugas sebagai pegawai di jabatan ini.",
]


def notice_sentence(rng: random.Random) -> Sentence:
    template = rng.choice(NOTICE_TEMPLATES).replace("{C}", _class_text(rng)).replace("{M}", str(rng.randint(10, 99))).replace("{Y}", str(rng.randint(2005, 2024)))
    upper = rng.random() < 0.12
    parts: list[Sentence] = []
    pos = 0
    for m in _SLOT.finditer(template):
        pre = template[pos:m.start()].strip()
        if pre:
            parts.append(_text(_up(pre) if upper else pre))
        kind, fn = _GEN[m.group(1)]
        toks = fn(rng)
        if upper:
            toks = [t.upper() for t in toks]
        parts.append(_ent(kind, toks, _trail(toks, m.group(2))))
        pos = m.end()
    tail = template[pos:].strip()
    if tail:
        parts.append(_text(_up(tail) if upper else tail))
    return _wrap(rng, _cat(parts), 0.3)


def form_block(rng: random.Random) -> Sentence:
    """'Label : value' lines. The label decides: Nama = PERSON; Nama Kelas, Kelas,
    Kod Sekolah, Pilihan Kerjaya, Umur ... = O although the value looks the same."""
    fields: list[Sentence] = []

    def label(txt: str) -> str:
        txt = _up(txt) if upper else txt
        return txt + (" :" if rng.random() < 0.6 else ":")

    upper = rng.random() < 0.4
    has_name = rng.random() < 0.8
    if has_name:
        toks = layer_person(rng)
        if upper:
            toks = [t.upper() for t in toks]
        fields.append(_cat([_text(label(rng.choice(["Nama", "Nama Murid", "Nama Pelajar", "Nama Penuh"]))), _ent("PERSON", toks)]))
    if rng.random() < 0.5:
        fields.append(_cat([_text(label(rng.choice(["No. KP", "No. MyKad", "No. Pengenalan", "No. ID"]))), _ent("NRIC", nric_tokens(rng))]))
    if rng.random() < 0.3:
        fields.append(_cat([_text(label("Alamat")), _ent("ADDRESS", layer_address(rng))]))
    others = [
        lambda: _text(f"{label('Kelas')} {_class_text(rng)}"),
        lambda: _text(f"{label('Nama Kelas')} {rng.choice(CLASS_ADJ)} {rng.choice(['DOMAIN', 'ZON', 'KUMPULAN', 'UNIT'])}"),
        lambda: _text(f"{label('Kod Sekolah')} {_school_code(rng)}"),
        lambda: _text(f"{label(rng.choice(['Sekolah', 'Institusi', 'Nama Institusi']))} {rng.choice(['SK', 'SMK', 'SJK', 'SEKOLAH RENDAH', 'KOLEJ'])} {_up(rng.choice(SCHOOL_PLACE))}"),
        lambda: _text(f"{label('Umur')} {rng.randint(6, 17)}"),
        lambda: _text(f"{label('Jantina')} {rng.choice(['Lelaki', 'Perempuan', 'LELAKI', 'PEREMPUAN'])}"),
        lambda: _text(f"{label('Pilihan Kerjaya')} {', '.join(rng.sample(JOBS, rng.randint(2, 4)))},"),
        lambda: _text(f"{label('Tahun')} {rng.randint(2020, 2026)}"),
        lambda: _text(f"{label('Skor')} {_title_phrase(rng)} - {_title_phrase(rng)}"),
    ]
    for fn in rng.sample(others, rng.randint(3, 6)):
        fields.append(fn())
    order = list(range(len(fields)))
    rng.shuffle(order)
    parts = [_text(rng.choice(["BORANG MAKLUMAT", "PROFIL MURID", "REKOD PELAJAR", "STUDENT PROFILE", "LAPORAN INDIVIDU"]))] \
        if rng.random() < 0.5 else []
    parts += [fields[i] for i in order]
    return _wrap(rng, _cat(parts), 0.3)


def name_between_heading_and_title(rng: random.Random) -> Sentence:
    toks = layer_person(rng)
    if rng.random() < 0.3:
        toks = [t.upper() for t in toks]
    parts = [_text(rng.choice(HEADINGS_PERSONAL))]
    if rng.random() < 0.4:
        parts.append(_text(rng.choice(["Dr.", "Dato'", "Prof.", "Tan Sri", "Haji"])))
    parts.append(_ent("PERSON", toks))
    parts.append(_text(rng.choice(TITLES_OFFICIAL)))
    if rng.random() < 0.5:
        parts.append(_text(rng.choice(["Kementerian Pendidikan Malaysia", "Jabatan Pendidikan Negeri", "Lembaga Peperiksaan", "Bahagian Sumber Manusia"])))
    return _wrap(rng, _cat(parts), 0.25)


def rural_address_line(rng: random.Random) -> Sentence:
    toks = layer_address(rng)
    lead = rng.choice(["beralamat di", "alamat tetap :", "Alamat surat-menyurat :", "tinggal di", "Address :", "residing at"])
    return _wrap(rng, _cat([_text(lead), _ent("ADDRESS", toks, _trail(toks, rng.choice([".", ""])))]), 0.3)


def student_list_rows(rng: random.Random) -> Sentence:
    parts = [_text(rng.choice(["BIL NAMA NO. K/P SEKOLAH KELAS", "NO NAMA MURID NO. PENGENALAN KELAS", "SENARAI MURID"]))]
    for i in range(1, rng.randint(3, 7)):
        toks = layer_person(rng)
        upper = rng.random() < 0.5
        if upper:
            toks = [t.upper() for t in toks]
        row = [_text(str(i)), _ent("PERSON", toks)]
        if rng.random() < 0.7:
            row.append(_ent("NRIC", nric_tokens(rng)))
        row.append(_text(f"SEKOLAH {rng.choice('ABCDEF')} {rng.randint(1, 6)}{rng.choice('ABCD')}"))
        parts.append(_cat(row))
    return _cat(parts)


POSITIVES = [(form_block, 0.12), (notice_sentence, 0.14), (rural_address_line, 0.05),
             (name_between_heading_and_title, 0.05), (student_list_rows, 0.08)]
FAMILIES = NEGATIVES + POSITIVES


def build_layer(rng: random.Random, count: int) -> list[Sentence]:
    funcs = [f for f, _ in FAMILIES]
    weights = [w for _, w in FAMILIES]
    return [rng.choices(funcs, weights)[0](rng) for _ in range(count)]


# ---------------------------------------------------------------------
# Checks and main
# ---------------------------------------------------------------------

def _valid(sent: Sentence) -> bool:
    if len(sent.tokens) != len(sent.labels) or not sent.tokens:
        return False
    prev = "O"
    for lab in sent.labels:
        if lab.startswith("I-") and (prev == "O" or prev[2:] != lab[2:]):
            return False
        prev = lab
    return True


def dedupe(sentences: list[Sentence]) -> list[Sentence]:
    seen: set[tuple] = set()
    out: list[Sentence] = []
    for s in sentences:
        key = (tuple(s.tokens), tuple(s.labels))
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def zero_entity_share(sentences: list[Sentence]) -> float:
    return sum(1 for s in sentences if all(l == "O" for l in s.labels)) / max(len(sentences), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v25", default="data/processed/semi_synthetic_v25_ns3k.txt")
    ap.add_argument("--layer-count", type=int, default=6500, help="new blocks (about 58 percent have no entity)")
    ap.add_argument("--seed", type=int, default=26)
    ap.add_argument("--out", default="data/processed/semi_synthetic_v26_ns3k.txt")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    raw = parse_iob2_file(args.v25)
    base = dedupe(raw)
    layer = dedupe(build_layer(rng, args.layer_count))
    bad = [s for s in layer if not _valid(s)]
    assert not bad, f"{len(bad)} invalid layer blocks, first: {bad[0].tokens[:20]} {bad[0].labels[:20]}"
    combined = base + layer
    rng.shuffle(combined)
    lens = sorted(len(s.tokens) for s in layer)
    print(f"v25 raw: {len(raw)} | after dedupe: {len(base)} | new layer: {len(layer)} "
          f"(len median {lens[len(lens) // 2]}, max {lens[-1]}) | total: {len(combined)}")
    print(f"zero-entity share: base {zero_entity_share(base):.1%} -> combined {zero_entity_share(combined):.1%} "
          f"(layer alone {zero_entity_share(layer):.1%})")
    before, after = entity_token_counts(base), entity_token_counts(combined)
    for tag in sorted(set(before) | set(after)):
        print(f"  {tag:8s} {before.get(tag, 0):7d} -> {after.get(tag, 0):7d}")
    write_iob2(combined, args.out)
    print(f"wrote {args.out}")
    print("next: python -m eval.dedupe_blocks <out> <dedup out>, then python -m eval.split_holdout ...")


if __name__ == "__main__":
    main()
