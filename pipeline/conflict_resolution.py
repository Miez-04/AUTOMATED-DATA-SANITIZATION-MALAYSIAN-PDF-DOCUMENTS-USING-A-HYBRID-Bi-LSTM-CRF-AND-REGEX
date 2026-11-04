"""
pipeline.conflict_resolution

Implements the FYP pseudocode's gate directly:

    FOR each token index ti:
        IF ti IN regex_mask_registry:
            label[ti] = regex_mask_registry.get(ti).label   # Tier 1 wins
        ELSE:
            label[ti] = tier2_prediction[ti]                # Tier 1 silent -> Tier 2

Edge case the pseudocode doesn't cover, found while testing this on
real multi-entity sentences: Tier 1 can force a token to B-NRIC/B-PHONE
in the *middle* of what Tier 2 predicted as a continuous I-PERSON /
I-ADDRESS span. That leaves a dangling "I-X" token whose preceding
token is no longer B-X or I-X of the same type — invalid IOB2, and it
would merge into the wrong entity in pipeline.coordinate_merge (or get
silently dropped depending on how strict the merge is). repair_iob2()
below is a required second pass, not optional cleanup: run it every
time, not just when something looks wrong.

Architecture decision (FYP report sections 3.2.1 and 3.3.1): the gate is
one deterministic registry lookup plus Tier 2's prediction. Tier 1 (regex)
covers NRIC and PHONE only; ADDRESS comes from Tier 2 only. Two rule-based
ADDRESS post-processors still exist in this file but are OFF by default and
can be switched on per request or per process, so they stay available for
ablation experiments (eval/ablate_combine.py):

    1. address heuristic   (_fill_address_gaps_with_heuristic)
       on : address_heuristic(True)  or env PDFSAN_ADDRESS_HEURISTIC=1
    2. address post-filter (_reject_implausible_model_address)
       on : address_postfilter(True) or env PDFSAN_ADDRESS_POSTFILTER=1

Measured with a perfect Tier 2 on the dedup holdout (1,047 gold ADDRESS
spans): both off = ADDRESS F1 0.9971 -> 1.0000 after the post-filter fixes
below, heuristic on = 0.9213.
"""

from __future__ import annotations

import contextlib
import contextvars
import os

from pdf_ingestion.schema import DocumentTokens, PageTokens
from regex_engine.address_heuristic import find_address_spans
from regex_engine.matcher import RegexMaskRegistry
from regex_engine.address_heuristic import (
    ADDRESS_ANCHOR_KEYWORDS,
    _STATE_TOKENS,
    _is_plausible_postcode,
)
from regex_engine.patterns import (
    NRIC_PATTERN,
    PHONE_PATTERNS,
    is_plausible_nric,
    is_plausible_phone,
)

MODEL_SOURCE = "model"
REGEX_SOURCE = "regex"
HEURISTIC_SOURCE = "address_heuristic"

# Minimum CRF marginal probability (see models.bilstm_crf.BiLSTMCRF.marginals
# / pipeline.inference.TierTwoPredictor.predict_page_with_confidence) a
# model-only PERSON/ADDRESS span must clear to survive into redaction.
#
# CURRENT VALUE 0 = NO automatic rejection, on purpose: FYP report 3.3.1
# says low-confidence and borderline Tier 2 predictions are not rejected
# at the gate; they are shown to the operator in the human-in-the-loop
# step (Step 4). The history below explains the earlier 0.5 / 0.7 values.
# NRIC/PHONE are excluded here: they have their own stricter, deterministic
# plausibility check below (_reject_implausible_model_nric_phone) that a
# probability threshold can't improve on.
#
# Raised from 0.5 to 0.7 after direct inspection via eval/dump_iob2.py
# and ui/static/iob2.html (which surface each token's real confidence
# number) showed irrelevant highlights clustering below 70% while
# genuine entities mostly cleared it — a concrete, inspected threshold
# rather than a guessed one. Started at 0.5 after real-PDF testing on
# an out-of-domain technical document showed the model producing
# widespread low-confidence guesses (hardware terms, part numbers,
# bare fragments) tagged as every entity type; a model has no real
# basis for confidence on content this far outside its training
# distribution, and this gate is what catches that regardless of how
# much more Malaysian-PII training data gets added. If this starts
# rejecting genuine entities too aggressively, check the actual
# confidence distribution again via the inspector before just
# lowering the number back — the right value is whatever the real
# data shows, not a round number.
MIN_MODEL_CONFIDENCE = 0


def resolve_page(
    page: PageTokens,
    registry: RegexMaskRegistry,
    tier2_labels: list[str],
    tier2_confidences: list[float] | None = None,
) -> None:
    """Mutates page.tokens' label/source/confidence in place. Tokens
    already labelled by regex_engine.matcher (source == "regex") are
    left untouched — that IS the "Tier 1 wins" rule, already applied at
    match time. Every other token gets its Tier 2 prediction.

    tier2_confidences is optional (defaults to 1.0/token) only so
    existing callers/tests that predate confidence scoring don't
    break — pipeline.run.sanitize_pdf should always pass real
    confidences now. Passing None disables
    _reject_low_confidence_model_predictions since there is nothing
    meaningful to threshold."""
    assert len(tier2_labels) == len(page.tokens), (
        f"tier2_labels length {len(tier2_labels)} != token count "
        f"{len(page.tokens)} for page {page.page_num}"
    )
    if tier2_confidences is None:
        tier2_confidences = [1.0] * len(page.tokens)
    assert len(tier2_confidences) == len(page.tokens), (
        f"tier2_confidences length {len(tier2_confidences)} != token count "
        f"{len(page.tokens)} for page {page.page_num}"
    )

    # NRIC/PHONE are regex-only by explicit decision — Tier 1's patterns
    # are comprehensive and already format-validated
    # (is_plausible_nric/is_plausible_phone in regex_engine.patterns),
    # so a model guess for these two types is never a genuine catch,
    # only a way to introduce a false positive the deterministic rules
    # would have rejected anyway. Discarded outright here (kept "O"),
    # not merely downgraded after the fact — the model is never even
    # allowed to claim these two labels for a token Tier 1 didn't
    # already claim itself.
    MODEL_VETOED_ENTITIES = {"NRIC", "PHONE"}

    for token, predicted_label, confidence in zip(
        page.tokens, tier2_labels, tier2_confidences
    ):
        if (page.page_num, token.token_index) in registry:
            continue  # Tier 1 already claimed this token; do not overwrite
        entity = predicted_label[2:] if predicted_label.startswith(("B-", "I-")) else None
        if entity in MODEL_VETOED_ENTITIES:
            continue  # discard the model's guess; token stays "O" (schema default)
        token.label = predicted_label
        token.source = MODEL_SOURCE
        token.confidence = confidence

    # Repair BEFORE the filters: the filters below only start a span on a
    # "B-" token, so a model span that begins with a dangling "I-X"
    # would skip every check and be converted to "B-X" afterwards.
    repair_iob2(page)
    if address_postfilter_enabled():  # OFF by default (report 3.3.1); on via address_postfilter(True) / env
        _reject_implausible_model_address(page)
    _reject_implausible_model_nric_phone(page)  # defensive no-op now for
                                                  # NRIC/PHONE (no model-
                                                  # sourced tokens of those
                                                  # types can exist), left
                                                  # in place in case that
                                                  # decision is ever revisited
    _reject_low_confidence_model_predictions(page)
    repair_iob2(page)
    _fill_address_gaps_with_heuristic(page)  # returns 0 immediately unless address_heuristic_enabled()


def _reject_low_confidence_model_predictions(
    page: PageTokens, min_confidence: float = MIN_MODEL_CONFIDENCE
) -> int:
    """Downgrades model-only PERSON/ADDRESS spans whose MINIMUM
    per-token confidence (weakest link in the span — one uncertain
    token is enough to make the whole span's boundary untrustworthy)
    falls below min_confidence. Restricted to PERSON/ADDRESS: NRIC/
    PHONE already have a stronger, format-based check that doesn't
    need this."""
    downgraded = 0
    i = 0
    n = len(page.tokens)
    while i < n:
        token = page.tokens[i]
        entity = token.label[2:] if token.label.startswith("B-") else None
        if token.source == MODEL_SOURCE and entity in ("PERSON", "ADDRESS"):
            span = [token]
            j = i + 1
            while (
                j < n
                and page.tokens[j].source == MODEL_SOURCE
                and page.tokens[j].label == f"I-{entity}"
            ):
                span.append(page.tokens[j])
                j += 1

            if min(t.confidence for t in span) < min_confidence:
                for t in span:
                    t.label = "O"
                downgraded += len(span)
            i = j
        else:
            i += 1
    return downgraded


# ---- address-heuristic on/off switch ------------------------------------
# Resolution order: (1) a per-request override set with address_heuristic(...)
# below (used by the web UI toggle), (2) the environment variable
# PDFSAN_NO_ADDRESS_HEURISTIC=1 (forces it off for the whole process),
# (3) the environment variable PDFSAN_ADDRESS_HEURISTIC=1 (turns it on for
# the whole process), (4) default: OFF. A context variable is used, not a
# global, so two requests handled at the same time cannot change each
# other's setting.
_ADDRESS_HEURISTIC_OVERRIDE: contextvars.ContextVar = contextvars.ContextVar(
    "address_heuristic_override", default=None
)


def address_heuristic_enabled() -> bool:
    """True if the address heuristic should run right now."""
    override = _ADDRESS_HEURISTIC_OVERRIDE.get()  # True / False / None (= no override)
    if override is not None:
        return override
    if os.environ.get("PDFSAN_NO_ADDRESS_HEURISTIC") == "1":
        return False
    return os.environ.get("PDFSAN_ADDRESS_HEURISTIC") == "1"  # default OFF


@contextlib.contextmanager
def address_heuristic(enabled: bool | None):
    """with address_heuristic(False): resolve_document(...)  -> heuristic off
    inside the block only. Passing None means 'no override' (env/default)."""
    token = _ADDRESS_HEURISTIC_OVERRIDE.set(enabled)
    try:
        yield
    finally:
        _ADDRESS_HEURISTIC_OVERRIDE.reset(token)


# ---- address post-filter on/off switch ----------------------------------
# Controls _reject_implausible_model_address inside resolve_page. Same
# resolution order idea as the heuristic switch: (1) per-call override set
# with address_postfilter(...), (2) env PDFSAN_ADDRESS_POSTFILTER=1,
# (3) default: OFF. Calling _reject_implausible_model_address(page)
# directly is never affected by this switch.
_ADDRESS_POSTFILTER_OVERRIDE: contextvars.ContextVar = contextvars.ContextVar(
    "address_postfilter_override", default=None
)


def address_postfilter_enabled() -> bool:
    """True if resolve_page should run the ADDRESS plausibility post-filter."""
    override = _ADDRESS_POSTFILTER_OVERRIDE.get()  # True / False / None (= no override)
    if override is not None:
        return override
    return os.environ.get("PDFSAN_ADDRESS_POSTFILTER") == "1"  # default OFF


@contextlib.contextmanager
def address_postfilter(enabled: bool | None):
    """with address_postfilter(True): resolve_page(...) -> post-filter on
    inside the block only. Passing None means 'no override' (env/default)."""
    token = _ADDRESS_POSTFILTER_OVERRIDE.set(enabled)
    try:
        yield
    finally:
        _ADDRESS_POSTFILTER_OVERRIDE.reset(token)


def _fill_address_gaps_with_heuristic(page: PageTokens) -> int:
    """
    Address heuristic (regex_engine.address_heuristic) runs LAST, as a
    pure gap-filler — NOT with NRIC/PHONE's "always wins" priority.

    Found necessary after an eval regression: giving the address
    heuristic the same absolute priority as NRIC/PHONE regex caused it
    to OVERRIDE already-correct model predictions whenever its own
    span boundary (anchor-keyword-based, approximate) differed by even
    one token from the model's — seqeval's exact-span matching then
    scored that as both a full miss AND a full false positive, despite
    most tokens agreeing. NRIC/PHONE regex is boundary-exact by
    construction (fixed digit-count patterns), so "always wins" is
    safe there; this heuristic is not, so it only fills spans that are
    CURRENTLY ENTIRELY "O" — i.e. genuine gaps the model missed
    completely (the real resume-PDF failure mode this was built for),
    never touching a token the model already made any prediction for.

    OFF by default (FYP report 3.2.1 / 3.3.1: ADDRESS is Tier 2's job).
    Measured with a perfect Tier 2: it lowered ADDRESS F1 from 0.9971 to
    0.9213 on the dedup holdout, because the span it writes can replace a
    correct model span with a longer one (extra leading registration
    numbers, trailing "No. Telefon:", merging across unrelated text).
    """
    if not address_heuristic_enabled():  # per-request UI switch / env switch
        return 0
    filled = 0
    token_texts = [t.text for t in page.tokens]
    for start_idx, end_idx in find_address_spans(token_texts):
        span = page.tokens[start_idx:end_idx]
        # Allow completing/correcting a PARTIAL address-only guess from
        # the model (found necessary: the model sometimes tags just 2
        # of a real 7-token address as ADDRESS with the wrong boundary,
        # which blocked the heuristic from supplying the full, correct
        # span under a strict "only touch tokens still O" rule) — but
        # still never override a DIFFERENT entity type the model
        # predicted, which is what the original override bug was about.
        if any(t.label != "O" and not t.label.endswith("ADDRESS") for t in span):
            continue

        # Does this heuristic span pick up right where an existing
        # ADDRESS entity (model- or regex-tagged) left off? If so,
        # CONTINUE it (I-ADDRESS) instead of starting a fresh one
        # (B-ADDRESS) — otherwise pipeline.coordinate_merge treats the
        # new B- as a separate entity, producing two adjacent
        # redaction boxes for what is really one address. Confirmed
        # via real-PDF testing: "No." tagged B-ADDRESS by the model,
        # then "55, Jalan Pasir Pinji 3, ..." filled by this heuristic
        # as its OWN B-ADDRESS-started span right after — same bug for
        # the reverse order (heuristic fills first, model's tagging
        # picks up immediately after and would otherwise re-B-ADDRESS).
        continues_before = (
            start_idx > 0 and page.tokens[start_idx - 1].label.endswith("ADDRESS")
        )

        for i, token in enumerate(span):
            if i == 0 and continues_before:
                token.label = "I-ADDRESS"
            else:
                token.label = "B-ADDRESS" if i == 0 else "I-ADDRESS"
            token.source = HEURISTIC_SOURCE
            filled += 1

        # Symmetric case: this heuristic span fills a gap immediately
        # BEFORE tokens the model/regex already tagged — that
        # following token is a continuation too now, not a new entity,
        # so it must not still be a "B-ADDRESS" once this span merges
        # into it.
        if end_idx < len(page.tokens) and page.tokens[end_idx].label == "B-ADDRESS":
            page.tokens[end_idx].label = "I-ADDRESS"

    return filled


# Extra anchors used ONLY by the plausibility check below, not by the
# heuristic's address finder, so the finder's false-positive guard for
# common words is unchanged.
_PLAUSIBILITY_EXTRA_ANCHORS = {"suite", "level", "aras"}


def _is_anchor_token(text: str) -> bool:
    """Input : text (str), one token such as "NO.", "Jalan," or "Pt.".
    Output: bool, True if the token is an address anchor word.

    Keeps a trailing "." so entries written with a period in
    ADDRESS_ANCHOR_KEYWORDS ("no.", "pt.", "lor.", "pers.") still match.
    The old check stripped the period first, so "NO." became "no", which
    is deliberately NOT in the set. Bare "no" stays excluded."""
    w = text.lower().strip(",;:()")  # str: lowercase, commas/brackets removed, period kept
    return (
        w in ADDRESS_ANCHOR_KEYWORDS
        or w.rstrip(".") in ADDRESS_ANCHOR_KEYWORDS
        or w in _PLAUSIBILITY_EXTRA_ANCHORS
    )


def _reject_implausible_model_address(page: PageTokens) -> int:
    """Cross-checks model-predicted ADDRESS spans against the SAME
    structural definition of a real Malaysian address that
    regex_engine.address_heuristic already uses to independently FIND
    addresses: a plausible postcode (5 digits, 1000-98999), a
    recognized state/federal-territory name, or a street/building-type
    anchor keyword (jalan, lot, taman, pangsapuri, ...).

    Added after ADDRESS F1 on the adversarial eval stayed stuck at
    exactly 0.6000 across four corpus iterations (v19-v22) despite
    different training-data strategies each time — a strong signal the
    model needed a deterministic structural check, the same lever that
    already worked for NRIC/PHONE, rather than more training data.
    Directly targets the reported failure: the model tagging bare
    numbers as ADDRESS with no postcode, city, or state anywhere in
    sight — something a real Malaysian address always has at least one
    of, per the same structural definition address_heuristic.py's own
    FINDER already relies on.

    Only downgrades MODEL-sourced tokens within a failing span —
    tokens from address_heuristic itself are address_heuristic-sourced
    and always satisfy this check anyway by construction (they're
    found FROM a postcode/anchor in the first place), so this adds a
    check only where one was missing, without duplicating work HITL
    already does downstream in the UI.
    """
    downgraded = 0
    i = 0
    n = len(page.tokens)
    while i < n:
        token = page.tokens[i]
        entity = token.label[2:] if token.label.startswith("B-") else None
        if entity == "ADDRESS":
            span = [token]
            j = i + 1
            while j < n and page.tokens[j].label == "I-ADDRESS":
                span.append(page.tokens[j])
                j += 1

            has_model_token = any(t.source == MODEL_SOURCE for t in span)
            if has_model_token:
                has_postcode = any(_is_plausible_postcode(t.text) for t in span)
                # Same normalisation the heuristic uses for states: strip
                # ",.;:()" first, so "Perak," matches "perak".
                has_state = any(
                    t.text.lower().strip(",.;:()") in _STATE_TOKENS for t in span
                )
                has_anchor = any(_is_anchor_token(t.text) for t in span)
                plausible = has_postcode or has_state or has_anchor

                if not plausible:
                    for t in span:
                        if t.source == MODEL_SOURCE:
                            t.label = "O"
                            downgraded += 1
            i = j
        else:
            i += 1
    return downgraded


def _reject_implausible_model_nric_phone(page: PageTokens) -> int:
    """Cross-checks NRIC/PHONE spans that include AT LEAST ONE
    model-predicted token against Tier 1's own format rules,
    downgrading the model-sourced tokens to "O" if the full span text
    fails plausibility.

    Checks every such span regardless of which token started it — not
    just spans where the FIRST token happens to be model-sourced.
    Found necessary: a span can start with a genuine Tier 1 regex
    match and then have the model tack on extra I-X continuation
    tokens (unrelated nearby digits/fragments) that were never
    independently verified — the original narrower check missed this
    entirely, since it only ever looked at spans STARTING at a
    model-sourced token.

    Only the model-sourced tokens within a failing span are downgraded
    — a regex-matched token is Tier 1 ground truth and stays labelled
    even if the model corrupted the span by extending it with
    implausible continuation; repair_iob2() (run right after this)
    cleans up whatever dangling I-X labels that leaves behind.

    Rationale: Tier 1's regex is comprehensive for NRIC/PHONE, so a
    span that fails Tier 1's own plausibility check is far more likely
    a false positive on unrelated digit-heavy content (student IDs,
    course codes, part numbers, years — confirmed via real test PDFs)
    than a genuine number Tier 1 somehow missed. Returns the number of
    tokens downgraded."""
    downgraded = 0
    i = 0
    n = len(page.tokens)
    while i < n:
        token = page.tokens[i]
        entity = token.label[2:] if token.label.startswith("B-") else None
        if entity in ("NRIC", "PHONE"):
            span = [token]
            j = i + 1
            while j < n and page.tokens[j].label == f"I-{entity}":
                span.append(page.tokens[j])
                j += 1

            has_model_token = any(t.source == MODEL_SOURCE for t in span)
            if has_model_token:
                span_text = " ".join(t.text for t in span)
                if entity == "NRIC":
                    m = NRIC_PATTERN.fullmatch(span_text)
                    plausible = bool(m and is_plausible_nric(m))
                else:
                    plausible = any(
                        (m := pattern.fullmatch(span_text)) and is_plausible_phone(m)
                        for pattern in PHONE_PATTERNS
                    )

                if not plausible:
                    for t in span:
                        if t.source == MODEL_SOURCE:
                            t.label = "O"
                            downgraded += 1
            i = j
        else:
            i += 1
    return downgraded


def repair_iob2(page: PageTokens) -> int:
    """Fixes dangling I-X tokens created when a regex hard-override
    lands inside a Tier-2-predicted span, or when Tier 2 itself emits
    an invalid sequence (BiLSTM-CRF with a full transition matrix can
    still do this on a genuinely uncertain token). A dangling I-X
    becomes B-X: it's still evidence of that entity type at that
    position, just no longer "continuing" the previous token's entity.
    Returns the number of tokens repaired (useful for eval/ logging)."""
    repaired = 0
    prev_label = "O"
    for token in page.tokens:
        label = token.label
        if label.startswith("I-"):
            entity = label[2:]
            valid_predecessor = prev_label in (f"B-{entity}", f"I-{entity}")
            if not valid_predecessor:
                token.label = f"B-{entity}"
                repaired += 1
        prev_label = token.label
    return repaired


def resolve_document(
    document: DocumentTokens,
    registry: RegexMaskRegistry,
    tier2_labels_by_page: list[list[str]],
    tier2_confidences_by_page: list[list[float]] | None = None,
) -> None:
    if tier2_confidences_by_page is None:
        tier2_confidences_by_page = [None] * len(document.pages)  # type: ignore[list-item]
    for page, tier2_labels, tier2_confidences in zip(
        document.pages, tier2_labels_by_page, tier2_confidences_by_page
    ):
        resolve_page(page, registry, tier2_labels, tier2_confidences)
