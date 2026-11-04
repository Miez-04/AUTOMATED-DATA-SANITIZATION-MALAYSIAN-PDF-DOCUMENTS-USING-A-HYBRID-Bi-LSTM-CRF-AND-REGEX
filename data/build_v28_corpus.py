"""
pdf-sanitizer/data/build_v28_corpus.py

Third layer, on top of the v27 corpus: TECHNICAL TEXT. Reason, measured on the
IOB2 dump of a 9,576-token group-project report (C++ / MPI code, formulas,
benchmark tables, references, a team table):

    PERSON  23 spans:  9 correct names,  13 code fragments,  1 phrase
        "N", "= N", "n =", "r =", "P =", "+= v *", "= (n *"  (variable, operator)
        "Basic Stats"  (Title Case metric name before a percentage)
        1 name has a role word fused to its front ("DEVELOPER MUHAMMAD ...")
    ADDRESS  9 spans:  9 false: comma-separated call arguments with numbers and
        UPPER_SNAKE macros ("&sum, 1, <TYPE>, <OP>, 0, <COMM>);") and one
        conference venue inside a reference ("<City>, NJ, USA, Apr.")
    NRIC / PHONE: 0 false (the regex tier ignored the 10-digit student IDs).

Why: in the v25 corpus only 71 tokens contain { } ; , 5 are operators and 3 are
call tokens, out of 538,145 tokens. Code is a kind of text the model never saw,
so its tokens carry no information and a lone capital letter next to "="
looks like the 118 "Dear Cik N," training spans (honorific + initial).

Layer content (label O unless an entity is named; the honorific + initial
positives stay in the corpus untouched):
  1. code lines        C/C++, CUDA, OpenMP, Python, Java/JS, SQL, shell, JSON, config
  2. definitions       "where N is the number of ...", "S = T1 / Tp", "P = 4", parameter tables
  3. results tables    rows of numbers, percentages, "Metric Name (88%), Other Metric (73%)"
  4. references        numbered citations: authors, title, "in Proc. ... Conf.", venue city, year, pages
  5. cover page        "Prepared for: NAME", "Lecturer: Dr. NAME", course codes, group lines    (NAME = PERSON)
  6. team table rows   NAME / student ID / ENGLISH ROLE in three column orders                  (NAME = PERSON)

Policy encoded here (state it in Chapter 3): authors and venues inside a numbered
reference are cited works, not data subjects, so they are labelled O.

Independence: no sentence, name or number is copied from a test document and no
MPI function names are used; the pools are generic. Torch-free.

    python -m data.build_v28_corpus ^
        --v27 data/processed/semi_synthetic_v27_ns3k.txt ^
        --out data/processed/semi_synthetic_v28_ns3k.txt

then: eval.dedupe_blocks, eval.split_holdout, make_unseen_test, models.train (as before).
"""

from __future__ import annotations

import argparse
import random

from data.build_v18_corpus import Sentence, entity_token_counts, parse_iob2_file, write_iob2
from data.build_v23_corpus import _cat
from data.build_v26_corpus import _ent, _text, _valid, dedupe, layer_person, zero_entity_share

# ---------------------------------------------------------------------
# Word banks (generic)
# ---------------------------------------------------------------------
LETTERS = list("nNpPrRiIjJkKmMqQtTxXyYzZ")
VARS = ["sum", "total", "count", "mean", "value", "result", "index", "offset", "limit", "buffer", "score", "delta",
        "alpha", "beta", "gamma", "rank", "size", "width", "height", "temp", "acc", "best", "prev", "next"]
ARRAYS = ["data", "values", "samples", "items", "matrix", "grid", "weights", "buf", "arr", "records"]
FUNCS = ["compute", "update", "reduce_all", "init_grid", "load_file", "normalize", "merge_sort", "scan_prefix",
         "launch_kernel", "gather_stats", "write_output", "check_bounds", "swap_rows", "apply_filter"]
TYPES = ["int", "double", "float", "long", "size_t", "unsigned", "char", "bool"]
MACROS = ["MAX_BUFFER_SIZE", "DEFAULT_TIMEOUT_MS", "NUM_WORKERS", "BLOCK_SIZE", "TILE_WIDTH", "OMP_NUM_THREADS",
          "CUDA_SUCCESS", "BUFFER_LEN", "HALO_WIDTH", "RETRY_LIMIT", "LOG_LEVEL_DEBUG", "PAGE_SIZE_BYTES"]
FILES = ["input", "results", "dataset", "config", "output", "readings", "scores", "metrics", "train", "report"]
TABLES = ["orders", "students", "readings", "events", "accounts", "sessions", "logs"]
COLUMNS = ["id", "name_code", "amount", "created_at", "status", "region", "score", "level", "duration"]
PROGRAMS = ["analyze", "simulate", "benchmark", "convert", "scheduler", "indexer", "renderer", "pipeline"]
PACKAGES = ["numpy", "pandas", "scipy", "matplotlib", "torch", "flask", "requests", "pytest"]
NOUNS = ["elements", "processes", "threads", "workers", "samples", "iterations", "nodes", "partitions", "records"]
PURPOSES = ["Measures baseline throughput", "Evaluates scaling behaviour", "Verifies the output checksum",
            "Compares memory usage", "Reports the average latency", "Checks load distribution"]
METRICS = ["Peak Finder", "Rolling Variance", "Cache Misses", "Load Balance", "Queue Depth", "Frame Rate", "Token Count",
           "Error Rate", "Memory Use", "Disk Writes", "Branch Hits", "Cold Start"]
SURNAMES = ["Fischer", "Okafor", "Lindqvist", "Moreau", "Tanaka", "Petrov", "Alvarez", "Novak", "Haddad", "Kowalski",
            "Eriksen", "Duarte", "Nakamura", "Brennan", "Castillo", "Varga"]
TITLE_WORDS = ["Scalable", "Efficient", "Parallel", "Adaptive", "Distributed", "Fast", "Robust", "Practical", "Lightweight",
               "Algorithms", "Methods", "Systems", "Analysis", "Framework", "Scheduling", "Aggregation", "Benchmarks"]
CONFS = ["Intl. Conf. High Perform. Comput.", "Symp. Parallel Process.", "Conf. Data Eng.", "Workshop Syst. Softw.",
         "Joint Comput. Conf.", "Annu. Meeting Comput. Sci."]
VENUES = ["Boston, MA, USA", "Denver, CO, USA", "Kyoto, Japan", "Lisbon, Portugal", "Sydney, NSW, Australia",
          "Austin, TX, USA", "Berlin, Germany", "Singapore", "Toronto, ON, Canada", "Seattle, WA, USA"]
MONTHS = ["Jan.", "Feb.", "Mar.", "Apr.", "May", "Jun.", "Jul.", "Aug.", "Sep.", "Oct.", "Nov.", "Dec."]
ROLES_EN = ["TEAM LEAD", "BACKEND DEVELOPER", "TESTER", "DOCUMENTATION LEAD", "SYSTEM ANALYST", "DATA ANALYST",
            "UI DESIGNER", "QA ENGINEER", "RESEARCH LEAD", "DEVOPS ENGINEER", "SCRUM MASTER"]
LECTURER_TITLES = ["Dr.", "Prof.", "Prof. Madya", "Ts.", "Puan", "Encik", "Dr", "Madam", "Sir"]
COVER_LABELS = ["Prepared for:", "Prepared by:", "Lecturer:", "Submitted to:", "Supervisor :", "Course Instructor :", "Presented to:"]


def _num(rng: random.Random) -> str:
    return rng.choice([str(rng.randint(1, 5000)), f"{rng.uniform(0, 100):.2f}", f"{rng.uniform(0, 1):.3f}", str(rng.choice([2, 4, 8, 16, 32, 64]))])


def _fill(template: str, rng: random.Random) -> str:
    """Replace the {slots} of a code template with random identifiers and numbers."""
    letter, v = rng.choice(LETTERS), rng.choice(VARS)
    repl = {
        "L": letter, "L2": rng.choice(LETTERS), "v": v, "v2": rng.choice(VARS), "arr": rng.choice(ARRAYS),
        "f": rng.choice(FUNCS), "T": rng.choice(TYPES), "M": rng.choice(MACROS), "M2": rng.choice(MACROS),
        "n": _num(rng), "n2": _num(rng), "file": rng.choice(FILES), "tbl": rng.choice(TABLES), "col": rng.choice(COLUMNS),
        "col2": rng.choice(COLUMNS), "prog": rng.choice(PROGRAMS), "pkg": rng.choice(PACKAGES), "p": rng.choice([1, 2, 4, 8, 16]),
        "d": "d_" + rng.choice(ARRAYS), "h": "h_" + rng.choice(ARRAYS),
    }
    return template.format(**repl)


CODE_TEMPLATES = [
    "for (int {L} = 0; {L} < {n}; ++{L}) {{ {v} += {arr}[{L}] * {arr}[{L}]; }}",
    "{T} {L} = static_cast<{T}>({arr}.size());",
    "{T} {v} = {L} * {L2} - {v2};",
    "if ({L} >= 0) {f}({arr}.data(), {L2}, {M}, {L}, 0, {M2});",
    "{f}({d}, {h}, {L} * sizeof(float), cudaMemcpyHostToDevice);",
    "#pragma omp parallel for reduction(+:{v})",
    "int main(int argc, char** argv) {{ {f}(&argc, &argv); return 0; }}",
    "std::vector<double> {arr}({n}, {n2});",
    "while ({v} < {M}) {{ {v} = {f}({v}, {L}); }}",
    "{L} = {n}; {L2} = {n2}; {v} += {L} * {L2};",
    "return {f}({arr}, {L}) / {L2};",
    "{v} = np.array([{n}, {n2}, {n}])",
    "def {f}({v}, {v2}=None): return {v} ** 2 + {L}",
    "for {L}, {v} in enumerate({arr}): print({L}, {v})",
    "df = pd.read_csv('{file}.csv'); df['{col}'].mean()",
    "const {v} = {arr}.map(({L}) => {L} * {n});",
    "public static void {f}(String[] args) {{ System.out.println({v}); }}",
    "SELECT {col}, COUNT(*) FROM {tbl} WHERE {col2} >= {n} GROUP BY {col};",
    '{{ "{col}": {n}, "{col2}": "{file}", "enabled": true }}',
    "{M}={n}",
    "./{prog}.exe --workers {p} --input data/{file}.csv > out/{file}.log",
    "make {prog} && ./{prog}.exe {p}",
    "pip install {pkg}",
    "git commit -m 'update {file}' && git push origin main",
    "{L}, {L2}, {n}, {M}, {n2}, {M2});",
    "&{v}, {n}, {M}, {M2}, 0, {M});",
]


def code_lines(rng: random.Random) -> Sentence:
    lines = [_fill(rng.choice(CODE_TEMPLATES), rng) for _ in range(rng.randint(1, 5))]
    lead = rng.choice(["", "", "Listing 1:", "Algorithm 2", "Code Snippet", "Example:"])
    return _text((lead + " " + " ".join(lines)).strip())


def definitions(rng: random.Random) -> Sentence:
    r = rng.random()
    if r < 0.30:
        a, b = rng.sample(LETTERS, 2)
        text = (f"where {a} is the number of {rng.choice(NOUNS)} and {b} is the number of {rng.choice(NOUNS)}, "
                f"{rng.choice(['and', 'while'])} {rng.choice(LETTERS)} = {_num(rng)}.")
    elif r < 0.55:
        text = " ".join(f"{rng.choice(LETTERS)} = {_num(rng)}{rng.choice([',', ';', ''])}" for _ in range(rng.randint(2, 4)))
    elif r < 0.80:
        rows = [f"{rng.choice(PROGRAMS)}.exe {rng.choice('PNTK')} = {rng.choice(['1', '2', '4', '2,3,4', '8'])} {rng.choice(PURPOSES)}"
                for _ in range(rng.randint(2, 4))]
        text = " ".join(rows)
    else:
        text = rng.choice(["S = T1 / Tp", "E = S / P", "T(n) = O(n log n)", "speedup = t_seq / t_par", "mean = sum / N",
                           "f(x) = a * x + b", "var = (1 / N) * sum(x_i - mean) ^ 2"])
        text = f"{rng.choice(['Formula', 'Equation', 'Eq.'])} ({rng.randint(1, 9)}) {text}"
    return _text(text)


def results_table(rng: random.Random) -> Sentence:
    r = rng.random()
    if r < 0.45:
        header = rng.choice(["Processes Time (s) Speedup Efficiency", "Threads Runtime (ms) Speedup", "Size Time Memory (MB)"])
        rows = []
        for k in range(rng.randint(3, 6)):
            rows.append(f"{2 ** k} {rng.uniform(0.5, 60):.2f} {rng.uniform(1, 9):.2f} {rng.randint(40, 100)}%")
        text = header + " " + " ".join(rows)
    elif r < 0.80:
        picks = rng.sample(METRICS, rng.randint(2, 4))
        text = (rng.choice(["followed closely by", "ranked highest:", "Top results:", "Overall,"]) + " " +
                ", ".join(f"{m} ({rng.randint(30, 99)}%)" for m in picks) + rng.choice([".", ""]))
    else:
        text = " ".join(f"{m} {rng.randint(1, 200)} {rng.randint(1, 99)}%" for m in rng.sample(METRICS, rng.randint(3, 5)))
    return _text(text)


def references(rng: random.Random) -> Sentence:
    items = []
    for k in range(1, rng.randint(2, 4)):
        authors = ", ".join(f"{rng.choice('ABCDEFGHJKLMNPRST')}. {rng.choice(SURNAMES)}" for _ in range(rng.randint(1, 3)))
        title = " ".join(rng.sample(TITLE_WORDS, rng.randint(3, 5)))
        a = rng.randint(10, 900)
        items.append(f"[{k}] {authors}, \"{title},\" in Proc. {rng.choice(CONFS)}, {rng.choice(VENUES)}, "
                     f"{rng.choice(MONTHS)} {rng.randint(1990, 2025)}, pp. {a}\u2013{a + rng.randint(2, 15)}.")
    return _text(" ".join(items))


def cover_page(rng: random.Random) -> Sentence:
    parts = []
    if rng.random() < 0.6:
        parts.append(_text(f"{rng.choice(['CSC', 'ITS', 'CSM', 'MAT'])}{rng.randint(100, 699)} {rng.choice(['GROUP', 'TEAM'])} {rng.randint(1, 12)} "
                           f"{rng.choice(['SEMESTER', 'TRIMESTER'])} {rng.randint(1, 6)} {rng.randint(2022, 2026)}"))
    for _ in range(rng.randint(1, 3)):
        toks = layer_person(rng)
        if rng.random() < 0.5:
            toks = [t.upper() for t in toks]
        parts.append(_text(rng.choice(COVER_LABELS)))
        if rng.random() < 0.4:
            parts.append(_text(rng.choice(LECTURER_TITLES)))
        parts.append(_ent("PERSON", toks))
    if rng.random() < 0.5:
        parts.append(_text(rng.choice(["ABSTRACT", "TABLE OF CONTENTS", "1. INTRODUCTION", "PROJECT REPORT"])))
    return _cat(parts)


def _student_id(rng: random.Random) -> str:
    return rng.choice([f"20{rng.randint(20, 26)}{rng.randint(100000, 999999)}", f"A{rng.randint(18, 25)}EC{rng.randint(1000, 9999)}",
                       f"S{rng.randint(10000, 99999)}"])


def team_table(rng: random.Random) -> Sentence:
    order = rng.choice(["name_id_role", "role_name_id", "id_name_role"])
    parts = [_text(rng.choice(["NO NAME STUDENT ID ROLE", "STUDENT ID ROLE", "TEAM MEMBERS", "NAME ID ROLE CONTRIBUTION"]))] \
        if rng.random() < 0.6 else []
    for _ in range(rng.randint(2, 5)):
        toks = layer_person(rng)
        if rng.random() < 0.7:
            toks = [t.upper() for t in toks]
        name = _ent("PERSON", toks)
        sid, role = _text(_student_id(rng)), _text(rng.choice(ROLES_EN))
        row = {"name_id_role": [name, sid, role], "role_name_id": [role, name, sid], "id_name_role": [sid, name, role]}[order]
        if rng.random() < 0.4:
            row.append(_text(f"{rng.choice([20, 25, 33, 50])}%"))
        parts.append(_cat(row))
    return _cat(parts)


FAMILIES = [(code_lines, 0.24), (definitions, 0.10), (results_table, 0.10), (references, 0.08), (cover_page, 0.20), (team_table, 0.28)]


def build_layer(rng: random.Random, count: int) -> list[Sentence]:
    funcs = [f for f, _ in FAMILIES]
    weights = [w for _, w in FAMILIES]
    return [rng.choices(funcs, weights)[0](rng) for _ in range(count)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v27", default="data/processed/semi_synthetic_v27_ns3k.txt")
    ap.add_argument("--layer-count", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=28)
    ap.add_argument("--out", default="data/processed/semi_synthetic_v28_ns3k.txt")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    raw = parse_iob2_file(args.v27)
    base = dedupe(raw)
    layer = dedupe(build_layer(rng, args.layer_count))
    bad = [s for s in layer if not _valid(s)]
    assert not bad, f"{len(bad)} invalid layer blocks, first: {bad[0].tokens[:20]} {bad[0].labels[:20]}"
    combined = base + layer
    rng.shuffle(combined)
    lens = sorted(len(s.tokens) for s in layer)
    print(f"v27 input: {len(raw)} | after dedupe: {len(base)} | new layer: {len(layer)} "
          f"(len median {lens[len(lens) // 2]}, max {lens[-1]}) | total: {len(combined)}")
    print(f"zero-entity share: base {zero_entity_share(base):.1%} -> combined {zero_entity_share(combined):.1%} "
          f"(layer alone {zero_entity_share(layer):.1%})")
    before, after = entity_token_counts(base), entity_token_counts(combined)
    for tag in sorted(set(before) | set(after)):
        print(f"  {tag:8s} {before.get(tag, 0):7d} -> {after.get(tag, 0):7d}")
    write_iob2(combined, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
