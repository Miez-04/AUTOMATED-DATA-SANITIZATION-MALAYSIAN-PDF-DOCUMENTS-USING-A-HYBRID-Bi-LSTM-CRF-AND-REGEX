"""
ui/server.py

Flask backend for the other PDF Sanitizer review UI (replaces the
Streamlit ui/app.py — see README.md's own note under requirements.txt:
"UI (HITL workspace) — swap for Flask+React later if needed").

Wraps the EXISTING pipeline unchanged:
    pdf_ingestion.PDFIngestor -> regex_engine.build_regex_mask_registry
    -> pipeline.inference.TierTwoPredictor.predict_document_with_confidence
    -> pipeline.conflict_resolution.resolve_document
    -> pipeline.coordinate_merge.merge_document_spans
    -> pipeline.redaction.apply_redactions

Nothing in pdf_ingestion/, regex_engine/, models/, or pipeline/ is
modified by this file — it only orchestrates them and serves the
result over HTTP for ui/static/index.html to render.

Run:
    pip install flask
    python -m ui.server
Then open http://127.0.0.1:5000 in a browser.
"""

from __future__ import annotations

import io
import os
import sys
import unicodedata
import tempfile
import uuid
from pathlib import Path

# Same reasoning as ui/app.py's bootstrap: pdf_ingestion/pipeline/
# regex_engine are siblings of this ui/ folder, not children of it —
# `python -m ui.server` from the project root already puts the root on
# sys.path, but this guards against running `python ui/server.py`
# directly instead, which wouldn't.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from flask import Flask, jsonify, request, send_file, send_from_directory

from pdf_ingestion.extractor import PDFIngestor
from pipeline.conflict_resolution import (
    _fill_address_gaps_with_heuristic,
    address_heuristic,
    address_heuristic_enabled,
    repair_iob2,
    resolve_document,
)
from pipeline.coordinate_merge import EntitySpan, _split_into_line_runs, merge_document_spans
from pipeline.inference import TierTwoPredictor
from pipeline.redaction import apply_redactions
from regex_engine.matcher import build_regex_mask_registry

app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"))

# The model the server uses when a request does not name one. Change it here, or
# without editing code: set PDFSAN_CHECKPOINT=models/checkpoints_vXX_ns/best_model.pt
# before starting the server. Both web pages follow this value.
DEFAULT_CHECKPOINT = os.environ.get("PDFSAN_CHECKPOINT", "models/checkpoints_v31_ns/best_model.pt")


def _checkpoint_from_form() -> str:
    """Checkpoint named in the request, or DEFAULT_CHECKPOINT when the field is
    missing or empty."""
    return (request.form.get("checkpoint") or "").strip() or DEFAULT_CHECKPOINT

# Whether the installed EntitySpan dataclass has a confidence field —
# only true if the confidence-scoring pipeline patch has been applied.
# Checked once so mask_word() below can construct EntitySpan correctly
# either way, matching the same getattr-based read in _span_to_dict.
import dataclasses as _dataclasses
_ENTITY_SPAN_HAS_CONFIDENCE = "confidence" in {f.name for f in _dataclasses.fields(EntitySpan)}

# In-memory session store. Fine for a single-user local FYP review
# tool; would need a real store (Redis, DB) for multi-user deployment.
DOCUMENTS: dict[str, dict] = {}

_predictor_cache: dict[str, TierTwoPredictor] = {}


def _get_predictor(checkpoint_path: str) -> TierTwoPredictor:
    if checkpoint_path not in _predictor_cache:
        _predictor_cache[checkpoint_path] = TierTwoPredictor(checkpoint_path)
    return _predictor_cache[checkpoint_path]


def _span_to_dict(span_id: str, span: EntitySpan, line_no: int) -> dict:
    # getattr, not span.confidence: EntitySpan only carries a confidence
    # field if the confidence-scoring pipeline patch has been applied.
    # Defaults to 1.0 (== "certain") on a plain/pristine pipeline so
    # this UI works either way rather than hard-requiring that patch.
    confidence = getattr(span, "confidence", 1.0)
    return {
        "id": span_id,
        "text": span.text,
        "type": span.entity_type,
        "page": span.page_num + 1,  # 1-indexed for the UI
        "line": line_no,
        # One rect PER VISUAL LINE the span crosses (see
        # pipeline.coordinate_merge.EntitySpan.bboxes) — a wrapped
        # multi-line address is 2+ boxes here, not one box spanning
        # both lines plus everything in between them.
        "bboxes": [
            {"x0": b.x0, "y0": b.y0, "x1": b.x1, "y1": b.y1} for b in span.bboxes
        ],
        "source": span.source,
        "confidence": round(confidence, 3),
    }


def _form_flag(name: str) -> bool | None:
    """Reads a true/false form field. Returns None when the field is missing,
    so the environment variable / default decides."""
    value = request.form.get(name)
    if value is None:
        return None
    return value.strip().lower() in {"true", "1", "on", "yes"}


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/config")
def config():
    """What the web pages need to show about the server: the model it uses by default."""
    return jsonify({"default_checkpoint": DEFAULT_CHECKPOINT})


@app.route("/api/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify({"error": "No file uploaded"}), 400

    checkpoint_path = _checkpoint_from_form()

    doc_id = uuid.uuid4().hex
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"pdfsan_{doc_id}_"))
    pdf_path = tmp_dir / "input.pdf"
    file.save(str(pdf_path))

    document = PDFIngestor().extract(str(pdf_path))
    predictor = _get_predictor(checkpoint_path)

    registry = build_regex_mask_registry(document)
    heuristic_flag = _form_flag("address_heuristic")  # True / False / None (= default)
    if hasattr(predictor, "predict_document_with_confidence"):
        tier2_labels_by_page, tier2_confidences_by_page = (
            predictor.predict_document_with_confidence(document)
        )
        with address_heuristic(heuristic_flag):
            heuristic_used = address_heuristic_enabled()
            resolve_document(document, registry, tier2_labels_by_page, tier2_confidences_by_page)
    else:
        # Plain/pristine pipeline (confidence-scoring patch not applied)
        # — predict_document/resolve_document only take labels.
        tier2_labels_by_page = predictor.predict_document(document)
        with address_heuristic(heuristic_flag):
            heuristic_used = address_heuristic_enabled()
            resolve_document(document, registry, tier2_labels_by_page)

    spans = merge_document_spans(document)

    # Approximate "line number" per span (mockup shows "pg 1 · line 3"):
    # rank of the span's top-edge (y0) among all spans on that page,
    # top-to-bottom. Not the PDF's real line index (would need
    # block/line metadata cross-referenced per span, which spans don't
    # currently carry) — an ordinal position readers can still use to
    # locate it on the page at a glance.
    spans_by_page: dict[int, list[EntitySpan]] = {}
    for s in spans:
        spans_by_page.setdefault(s.page_num, []).append(s)
    line_no_by_span_index: dict[int, int] = {}
    for page_num, page_spans in spans_by_page.items():
        ordered = sorted(range(len(page_spans)), key=lambda i: page_spans[i].bbox.y0)
        for rank, idx in enumerate(ordered, start=1):
            line_no_by_span_index[id(page_spans[idx])] = rank

    entities = []
    span_registry: dict[str, EntitySpan] = {}
    for span in spans:
        span_id = uuid.uuid4().hex
        span_registry[span_id] = span
        entities.append(_span_to_dict(span_id, span, line_no_by_span_index[id(span)]))

    DOCUMENTS[doc_id] = {
        "pdf_path": pdf_path,
        "tmp_dir": tmp_dir,
        "document": document,
        "spans_by_id": span_registry,
        "excluded_ids": set(),
        "filename": file.filename,
    }

    return jsonify({
        "doc_id": doc_id,
        "filename": file.filename,
        "num_pages": document.num_pages,
        "address_heuristic": heuristic_used,
        "pages": [
            {"page": p.page_num + 1, "width": p.page_width, "height": p.page_height}
            for p in document.pages
        ],
        "entities": entities,
    })


@app.route("/api/page/<doc_id>/<int:page_num>.png")
def page_image(doc_id: str, page_num: int):
    """Renders the ORIGINAL (unredacted) page as PNG for review-mode
    display — the frontend overlays entity boxes on top of this via
    CSS, matching the mockup's approach of showing labels layered on
    the real document rather than a pre-redacted image, so toggling
    an entity on/off doesn't require re-rendering the page."""
    entry = DOCUMENTS.get(doc_id)
    if entry is None:
        return "Not found", 404

    import fitz  # local import: keeps the flask process importable
                 # even in an environment where fitz isn't needed yet
    zoom = float(request.args.get("zoom", 1.6))
    doc = fitz.open(str(entry["pdf_path"]))
    try:
        page = doc[page_num - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        buf = io.BytesIO(pix.tobytes("png"))
    finally:
        doc.close()
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


def _build_entity_span(tokens: list, entity_type: str, page_num: int, source: str) -> EntitySpan:
    """Shared by mask_word and manual_select: given an ordered list of
    tokens (already sorted by token_index — reading order), builds the
    EntitySpan with correct per-line bboxes via the same splitting
    logic pipeline.coordinate_merge.merge_page_spans uses for model/
    regex-detected entities, so a manually-selected or other-matched
    span behaves identically to an auto-detected one everywhere
    downstream (review UI, redaction)."""
    bbox = tokens[0].bbox
    for t in tokens[1:]:
        bbox = bbox.union(t.bbox)
    line_runs = _split_into_line_runs(tokens)
    bboxes = []
    for run in line_runs:
        run_bbox = run[0].bbox
        for t in run[1:]:
            run_bbox = run_bbox.union(t.bbox)
        bboxes.append(run_bbox)
    span_kwargs = dict(
        entity_type=entity_type,
        text=" ".join(t.text for t in tokens),
        bbox=bbox,
        bboxes=tuple(bboxes),
        page_num=page_num,
        token_indices=tuple(t.token_index for t in tokens),
        source=source,
    )
    if _ENTITY_SPAN_HAS_CONFIDENCE:
        span_kwargs["confidence"] = 1.0
    return EntitySpan(**span_kwargs)


import re

_LEADER_DECORATION_PATTERN = re.compile(r"^[.\-_·…]{2,}$")


def _is_leader_decoration(text: str) -> bool:
    """True for signature-line / form-field leader decorations like
    '...................', '________', or '---' — tokens made up
    entirely of repeated punctuation with no real content. These
    should never end up inside a selected PERSON/NRIC/PHONE/ADDRESS
    span even when a drag's rectangle geometrically covers them (a
    drag over a name that sits right after a leader-dots run on the
    same line will often graze the dots too)."""
    return bool(_LEADER_DECORATION_PATTERN.match(text.strip()))


def _bbox_center_inside(token_bbox, x0: float, y0: float, x1: float, y1: float) -> bool:
    """A token counts as 'selected' only if its CENTER point falls
    inside the drag rectangle — not merely any overlap. Plain overlap
    was catching decorations like signature-line underscores or a
    ruled line sitting just below the intended word, since those only
    need to graze the edge of the drag box to match; requiring the
    center point makes selection match what the user actually dragged
    over, not what merely touches the box."""
    cx = (token_bbox.x0 + token_bbox.x1) / 2
    cy = (token_bbox.y0 + token_bbox.y1) / 2
    return x0 <= cx <= x1 and y0 <= cy <= y1


ALLOWED_MASK_TYPES = {"PERSON", "NRIC", "PHONE", "ADDRESS", "OTHER"}


def _norm_search(text: str) -> str:
    """Input : any text (str). Output: str with only lower-case letters and digits.
    Quotes, dashes, slashes, commas, brackets and spaces are removed, so
    "Rahmat," matches "Rahmat", "950325 - 10 - 6124" matches "950325-10-6124",
    and "SA\u2019DUN" matches "SA'DUN"."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(ch for ch in text if ch.isalnum())


def find_phrase_token_ranges(token_texts: list[str], phrase: str):
    """Input : token_texts (list of str, one page in reading order), phrase (str).
    Output: (ranges, inside_token) where ranges is a list of (first_index, last_index)
    token ranges, last_index inclusive, and inside_token is True when the matches were
    found only INSIDE a longer token.

    Step 1: strip every token and the phrase down to letters and digits (see _norm_search)
    and join the tokens into one string, so a name split over several tokens, a MyKad
    split into "950325 - 10 - 6124" or a phone split as "012-345 6789" is still one run
    of characters. Step 2: find the phrase in that string. A match counts when it starts
    where a token starts and ends where a token ends. Step 3: only when nothing matched
    that way, accept a match that lies entirely inside ONE token ("Maybank" inside
    "Menara-Maybank,"), so a search never invents a match across two words."""
    target = _norm_search(phrase)
    if not target:
        return [], False
    joined, owner, starts, ends = [], [], {}, {}
    for idx, text in enumerate(token_texts):
        norm = _norm_search(text)
        if not norm:
            continue                                   # pure punctuation token: contributes no characters
        starts[idx] = len("".join(joined))
        joined.append(norm)
        owner.extend([idx] * len(norm))
        ends[idx] = len("".join(joined))
    hay = "".join(joined)                              # str: all token texts, letters and digits only

    ranges, pos = [], 0
    while True:
        at = hay.find(target, pos)
        if at < 0:
            break
        first, last = owner[at], owner[at + len(target) - 1]
        if starts[first] == at and ends[last] == at + len(target):
            ranges.append((first, last))
            pos = at + len(target)                     # matches never overlap
        else:
            pos = at + 1
    if ranges:
        return ranges, False

    inside, pos = [], 0
    while True:
        at = hay.find(target, pos)
        if at < 0:
            break
        first, last = owner[at], owner[at + len(target) - 1]
        if first == last:                              # entirely inside one token
            inside.append((first, last))
            pos = ends[last]
        else:
            pos = at + 1
    return inside, bool(inside)


def merge_manual_match(page, spans_by_id: dict, first: int, last: int, entity_type: str):
    """Input : page (PageTokens), spans_by_id (dict id -> EntitySpan, edited in place),
    first / last (int, token positions of one manual match, last inclusive), entity_type (str).
    Output: (new_span or None, removed_ids list of str, outcome str).

    Rule: a manual match only ADDS or WIDENS. It never makes a highlight smaller, so it can
    never un-mask something that was already masked.
      no highlight overlaps the match                  -> "added"
      same tokens, same type                           -> "covered"   (nothing changes)
      an existing highlight is larger than the match   -> "covered"   (keep the larger one)
      same tokens, different type                      -> "retyped"   (the type you chose wins)
      match is larger than, or overlaps, a highlight   -> "extended"  (old highlight replaced by the union)
    To make a highlight smaller, use Reselect or delete it."""
    match_tokens = page.tokens[first:last + 1]
    wanted = {t.token_index for t in match_tokens}                    # set of int: token indices of the match
    overlapping = [(sid, sp) for sid, sp in spans_by_id.items()
                   if sp.page_num == page.page_num and wanted & set(sp.token_indices)]
    if not overlapping:
        return _build_entity_span(match_tokens, entity_type, page.page_num, source="manual"), [], "added"
    for _sid, sp in overlapping:
        existing = set(sp.token_indices)
        if wanted < existing:                                         # existing highlight is strictly larger
            return None, [], "covered"
    if len(overlapping) == 1:
        sid, sp = overlapping[0]
        if set(sp.token_indices) == wanted:
            if sp.entity_type == entity_type:
                return None, [], "covered"
            return _build_entity_span(match_tokens, entity_type, page.page_num, source="manual"), [sid], "retyped"
    union = set(wanted)
    for _sid, sp in overlapping:
        union |= set(sp.token_indices)
    union_tokens = sorted((t for t in page.tokens if t.token_index in union), key=lambda t: t.token_index)
    span = _build_entity_span(union_tokens, entity_type, page.page_num, source="manual")
    return span, [sid for sid, _ in overlapping], "extended"


@app.route("/api/mask-word", methods=["POST"])
def mask_word():
    """'Mask your own words' — exact (case-insensitive) phrase match
    against each page's reconstructed text, across every page. Reuses
    the SAME token list already extracted at upload time (no
    re-ingestion), so bboxes line up with what's already on screen.
    entity_type lets the caller tag the match as one of the real
    PERSON/NRIC/PHONE/ADDRESS types (so it merges into the existing
    filter pills/counts) rather than always being a separate OTHER
    bucket; defaults to OTHER only if omitted."""
    data = request.get_json(force=True)
    doc_id = data.get("doc_id")
    phrase = (data.get("phrase") or "").strip()
    entity_type = (data.get("entity_type") or "OTHER").strip().upper()
    entry = DOCUMENTS.get(doc_id)
    if entry is None:
        return jsonify({"error": "Unknown doc_id"}), 404
    if not phrase:
        return jsonify({"error": "Empty phrase"}), 400
    if entity_type not in ALLOWED_MASK_TYPES:
        return jsonify({"error": f"Invalid entity_type: {entity_type}"}), 400

    document = entry["document"]
    if not _norm_search(phrase):
        return jsonify({"error": "The phrase has no letters or digits"}), 400
    new_entities = []
    removed_ids: list[str] = []
    counts = {"added": 0, "extended": 0, "retyped": 0, "covered": 0}
    inside_any = False

    for page in document.pages:
        ranges, inside = find_phrase_token_ranges([t.text for t in page.tokens], phrase)
        inside_any = inside_any or inside
        for first, last in ranges:
            span, removed, outcome = merge_manual_match(page, entry["spans_by_id"], first, last, entity_type)
            counts[outcome] += 1
            for sid in removed:
                entry["spans_by_id"].pop(sid, None)
                removed_ids.append(sid)
            if span is not None:
                span_id = uuid.uuid4().hex
                entry["spans_by_id"][span_id] = span
                new_entities.append(_span_to_dict(span_id, span, line_no=0))

    return jsonify({"entities": new_entities, "removed_ids": removed_ids, "counts": counts,
                    "inside_token": inside_any})


@app.route("/api/manual-select", methods=["POST"])
def manual_select():
    """Drag-to-select on the page image. The frontend sends the
    dragged rectangle already converted to PDF-point coordinates
    (using the same scale factors it uses to position overlays), not
    raw screen pixels. Every token whose bbox overlaps that rectangle
    is picked up, sorted into reading order, and built into one
    EntitySpan the same way an auto-detected or mask-word entity is —
    including the per-line split, so a selection dragged across
    multiple lines becomes one attribute in the list but redacts as
    separate per-line boxes, not one rectangle spanning the gap."""
    data = request.get_json(force=True)
    doc_id = data.get("doc_id")
    page_num_1indexed = data.get("page")
    entity_type = (data.get("entity_type") or "OTHER").strip().upper()
    entry = DOCUMENTS.get(doc_id)
    if entry is None:
        return jsonify({"error": "Unknown doc_id"}), 404
    if entity_type not in ALLOWED_MASK_TYPES:
        return jsonify({"error": f"Invalid entity_type: {entity_type}"}), 400
    try:
        x0, y0, x1, y1 = (float(data[k]) for k in ("x0", "y0", "x1", "y1"))
        page_num_1indexed = int(page_num_1indexed)
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "x0/y0/x1/y1/page must be numbers"}), 400
    if x1 <= x0 or y1 <= y0:
        return jsonify({"error": "Empty selection"}), 400

    document = entry["document"]
    page_num_0indexed = page_num_1indexed - 1
    if not (0 <= page_num_0indexed < len(document.pages)):
        return jsonify({"error": "page out of range"}), 400
    page = document.pages[page_num_0indexed]

    selected_tokens = sorted(
        (t for t in page.tokens if _bbox_center_inside(t.bbox, x0, y0, x1, y1) and not _is_leader_decoration(t.text)),
        key=lambda t: t.token_index,
    )
    if not selected_tokens:
        return jsonify({"error": "No text found under that selection"}), 400

    span = _build_entity_span(selected_tokens, entity_type, page.page_num, source="manual")
    span_id = uuid.uuid4().hex
    entry["spans_by_id"][span_id] = span
    return jsonify({"entity": _span_to_dict(span_id, span, line_no=0)})


@app.route("/api/entity/<doc_id>/<entity_id>", methods=["DELETE"])
def delete_entity(doc_id: str, entity_id: str):
    """Fully remove a manually-added (or auto-detected) entity from
    the list — distinct from /api/toggle, which only excludes it from
    the final redaction while keeping it visible/re-includable."""
    entry = DOCUMENTS.get(doc_id)
    if entry is None or entity_id not in entry["spans_by_id"]:
        return jsonify({"error": "Unknown doc_id/entity_id"}), 404
    del entry["spans_by_id"][entity_id]
    entry["excluded_ids"].discard(entity_id)
    return jsonify({"ok": True})


@app.route("/api/retag/<doc_id>/<entity_id>", methods=["POST"])
def retag_entity(doc_id: str, entity_id: str):
    """Change an existing entity's type in place (same id, same
    tokens/bboxes) — e.g. the model tagged something as ADDRESS but it
    was actually a PHONE."""
    entry = DOCUMENTS.get(doc_id)
    if entry is None or entity_id not in entry["spans_by_id"]:
        return jsonify({"error": "Unknown doc_id/entity_id"}), 404
    data = request.get_json(force=True)
    new_type = (data.get("entity_type") or "").strip().upper()
    if new_type not in ALLOWED_MASK_TYPES:
        return jsonify({"error": f"Invalid entity_type: {new_type}"}), 400
    old_span = entry["spans_by_id"][entity_id]
    new_span = _dataclasses.replace(old_span, entity_type=new_type)
    entry["spans_by_id"][entity_id] = new_span
    return jsonify({"entity": _span_to_dict(entity_id, new_span, line_no=0)})


@app.route("/api/reselect-entity", methods=["POST"])
def reselect_entity():
    """Completely REPLACES an existing entity's token set with a fresh
    drag selection. Use this when the current highlight is wrong in
    either direction: too big (grabbed extra text that isn't part of
    the attribute) or too small. Keeps the same entity id (same list
    row); entity_type can optionally change too."""
    data = request.get_json(force=True)
    doc_id = data.get("doc_id")
    entity_id = data.get("entity_id")
    entry = DOCUMENTS.get(doc_id)
    if entry is None or entity_id not in entry["spans_by_id"]:
        return jsonify({"error": "Unknown doc_id/entity_id"}), 404
    try:
        x0, y0, x1, y1 = (float(data[k]) for k in ("x0", "y0", "x1", "y1"))
        page_num_1indexed = int(data["page"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "x0/y0/x1/y1/page must be numbers"}), 400

    existing_span = entry["spans_by_id"][entity_id]
    entity_type = (data.get("entity_type") or existing_span.entity_type).strip().upper()
    if entity_type not in ALLOWED_MASK_TYPES:
        return jsonify({"error": f"Invalid entity_type: {entity_type}"}), 400

    document = entry["document"]
    page_num_0indexed = page_num_1indexed - 1
    if not (0 <= page_num_0indexed < len(document.pages)):
        return jsonify({"error": "page out of range"}), 400
    page = document.pages[page_num_0indexed]

    new_tokens = sorted(
        (t for t in page.tokens
         if _bbox_center_inside(t.bbox, x0, y0, x1, y1) and not _is_leader_decoration(t.text)),
        key=lambda t: t.token_index,
    )
    if not new_tokens:
        return jsonify({"error": "No text found under that selection"}), 400

    new_span = _build_entity_span(new_tokens, entity_type, page.page_num, source="manual")
    entry["spans_by_id"][entity_id] = new_span  # same id -> replaces this row entirely, not merged
    return jsonify({"entity": _span_to_dict(entity_id, new_span, line_no=0)})


@app.route("/api/toggle/<doc_id>/<entity_id>", methods=["POST"])
def toggle_entity(doc_id: str, entity_id: str):
    """Review-before-download: exclude/re-include one detected span
    from the final redaction pass without deleting it from the list."""
    entry = DOCUMENTS.get(doc_id)
    if entry is None or entity_id not in entry["spans_by_id"]:
        return jsonify({"error": "Unknown doc_id/entity_id"}), 404
    data = request.get_json(force=True)
    include = bool(data.get("include", True))
    if include:
        entry["excluded_ids"].discard(entity_id)
    else:
        entry["excluded_ids"].add(entity_id)
    return jsonify({"ok": True})


@app.route("/api/sanitize/<doc_id>", methods=["POST"])
def sanitize(doc_id: str):
    entry = DOCUMENTS.get(doc_id)
    if entry is None:
        return jsonify({"error": "Unknown doc_id"}), 404

    active_spans = [
        span for sid, span in entry["spans_by_id"].items()
        if sid not in entry["excluded_ids"]
    ]
    out_path = entry["tmp_dir"] / "sanitized.pdf"
    count = apply_redactions(str(entry["pdf_path"]), str(out_path), active_spans)
    entry["output_path"] = out_path
    return jsonify({"ok": True, "redacted_count": count})


@app.route("/api/download/<doc_id>")
def download(doc_id: str):
    entry = DOCUMENTS.get(doc_id)
    if entry is None or "output_path" not in entry:
        return "Not sanitized yet", 404
    download_name = f"sanitized_{entry['filename']}"
    return send_file(str(entry["output_path"]), as_attachment=True, download_name=download_name)


@app.route("/iob2")
def iob2_page():
    return send_from_directory(app.static_folder, "iob2.html")


@app.route("/api/iob2-dump", methods=["POST"])
def iob2_dump():
    """Web equivalent of eval/dump_iob2.py — same pipeline, same
    token/label/source/confidence output, just returned as JSON for
    ui/static/iob2.html to render instead of printed to a terminal."""
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify({"error": "No file uploaded"}), 400

    checkpoint_path = _checkpoint_from_form()
    no_model = request.form.get("no_model") == "true"

    tmp_dir = Path(tempfile.mkdtemp(prefix="iob2dump_"))
    pdf_path = tmp_dir / "input.pdf"
    file.save(str(pdf_path))

    document = PDFIngestor().extract(str(pdf_path))
    registry = build_regex_mask_registry(document)

    heuristic_flag = _form_flag("address_heuristic")  # True / False / None (= default)
    with address_heuristic(heuristic_flag):
        heuristic_used = address_heuristic_enabled()
        if no_model:
            # Genuine Tier-1-only pass — see eval/dump_iob2.py's comment on
            # why this isn't done by faking an all-O tier2 prediction.
            for page in document.pages:
                _fill_address_gaps_with_heuristic(page)
                repair_iob2(page)
        else:
            predictor = _get_predictor(checkpoint_path)
            if hasattr(predictor, "predict_document_with_confidence"):
                labels_by_page, confidences_by_page = predictor.predict_document_with_confidence(document)
            else:
                labels_by_page = predictor.predict_document(document)
                confidences_by_page = [[1.0] * len(p.tokens) for p in document.pages]
            resolve_document(document, registry, labels_by_page, confidences_by_page)

    pages_out = []
    for page in document.pages:
        tokens_out = [
            {
                "text": t.text,
                "label": t.label,
                "source": t.source or None,
                "confidence": round(t.confidence, 3) if t.source == "model" else None,
                "block": t.block_no,
                "line": t.line_no,
            }
            for t in page.tokens
        ]
        pages_out.append({"page": page.page_num + 1, "tokens": tokens_out})

    return jsonify({
        "filename": file.filename,
        "address_heuristic": heuristic_used,
        "checkpoint": None if no_model else checkpoint_path,  # model actually used (None = Tier 1 only)
        "pages": pages_out,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
