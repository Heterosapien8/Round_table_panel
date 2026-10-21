"""parsing.py -- turn raw model text into structured turns (panelist, adversary) and auditor verdicts.
No silent defaults: anything missing/malformed -> None + parse_errors."""
import json
import re

from candidate_profiles_upgraded import CANDIDATE_LIST
from prompts_adversary import PERSONA_MECHANISMS
from verification import (NUM_RE, verify_fact, verify_prose_scores, verify_scores)

TAGS = ["SCORES", "CLAIM", "RANKING", "CITES_EVIDENCE", "CITED_FIELD", "CONFIDENCE",
        "SUMMARY_INJECTION", "OMITTED", "MISCHARACTERIZED_AGENT"]
_TAG_ALT = "|".join(TAGS)
_STRIP_CHARS = " \t\r\n[]()\"'`*"
FACT_RE = re.compile(r"^\s*(C\d+)\s*[.:]\s*([A-Za-z_]+)\s*=\s*(.+?)\s*$", re.IGNORECASE)


def strip_reasoning(text):
    """Drop <think>...</think> blocks emitted by reasoning models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def extract_tag(text, tag):
    """Capture colon-style or XML-style tags, preserving multi-line values."""
    m = re.search(
        rf"<{tag}>\s*:?\s*(.*?)(?=</{tag}>|<(?:{_TAG_ALT})>|\Z)",
        text, re.DOTALL | re.IGNORECASE,
    )
    return m.group(1).strip() if m else None


def _clean(v):
    return v.strip().strip(_STRIP_CHARS)


def parse_bool(v):
    if v is None:
        return None
    s = _clean(v).lower().rstrip(".!")
    if s == "true":
        return True
    if s == "false":
        return False
    return None  # includes the literal placeholder "true or false"


def parse_confidence(v):
    if v is None:
        return None
    nums = re.findall(NUM_RE, v)
    if len(nums) != 1:  # rejects "0.0 to 1.0" placeholder copies
        return None
    x = float(nums[0])
    return round(x, 3) if 0.0 <= x <= 1.0 else None


def parse_ranking(v):
    """Returns (ranking, valid). Valid only if it is an exact permutation of all candidates."""
    if v is None:
        return [], False
    ids = re.findall(r"\bC\d+\b", v.upper())
    valid = len(ids) == len(CANDIDATE_LIST) and set(ids) == set(CANDIDATE_LIST)
    return (ids if valid else []), valid


def parse_cited_facts(raw):
    """'C5.headline=...; C1.projects=...' -> (facts, unparsed_entries)"""
    if raw is None:
        return [], []
    s = _clean(raw)
    if s.lower() in ("", "none", "n/a"):
        return [], []
    parts = [p for p in re.split(r"\n|\s*\|\s*|,\s*(?=C\d+\s*[.:])", s, flags=re.IGNORECASE) if p.strip()]
    facts, unparsed = [], []
    for p in parts:
        m = FACT_RE.match(_clean(p))
        if not m:
            unparsed.append(_clean(p))
        else:
            facts.append({
                "candidate_id": m.group(1).upper(),
                "field": m.group(2).lower(),
                "claimed_value": _clean(m.group(3)),
            })
    return facts, unparsed


def parse_scores(v):
    """'C1=0.744; C2=0.715; ...' -> (scores dict, None) or (None, error)."""
    if v is None:
        return None, "missing_SCORES"
    pairs = re.findall(r"\b(C\d+)\s*[=:]\s*(-?\d*\.?\d+)", v.upper())
    ids = [p[0] for p in pairs]
    if len(ids) != len(CANDIDATE_LIST) or set(ids) != set(CANDIDATE_LIST):
        return None, "bad_SCORES"
    scores = {cid: float(x) for cid, x in pairs}
    if any(not (0.0 <= s <= 1.0) for s in scores.values()):
        return None, "bad_SCORES"
    return scores, None


def parse_panelist_output(raw_text, agent_id, require_scores=True):
    text = strip_reasoning(raw_text)
    errors = []

    claim = extract_tag(text, "CLAIM")
    if not claim:
        errors.append("missing_or_empty_CLAIM")
        claim = ""

    cites_raw = extract_tag(text, "CITES_EVIDENCE")
    cites = parse_bool(cites_raw)
    if cites is None:
        errors.append("bad_CITES_EVIDENCE")

    cited_raw = extract_tag(text, "CITED_FIELD")
    if cited_raw is None:
        errors.append("missing_CITED_FIELD")
    facts, unparsed = parse_cited_facts(cited_raw)
    if unparsed:
        errors.append(f"unparseable_cited_entries:{unparsed}")
    if cites is True and not facts:
        errors.append("cites_true_but_no_parseable_facts")
    if cites is False and facts:
        errors.append("cites_false_but_facts_listed")

    ranking, ranking_valid = parse_ranking(extract_tag(text, "RANKING"))
    if not ranking_valid:
        errors.append("invalid_RANKING")

    confidence = parse_confidence(extract_tag(text, "CONFIDENCE"))
    if confidence is None:
        errors.append("bad_CONFIDENCE")

    raw_scores = extract_tag(text, "SCORES")
    scores, score_err = parse_scores(raw_scores)
    if score_err and (require_scores or raw_scores is not None):
        errors.append(score_err)
    score_check = verify_scores(scores, ranking, ranking_valid) if scores else None

    detail = [verify_fact(f) for f in facts]
    prose_scores = verify_prose_scores(claim)
    # Tier 1 = everything deterministically checkable must check out:
    #   (a) cited raw profile values, (b) weighted scores stated in the prose, (c) the SCORES tag.
    fields_ok = None if not detail else all(d["status"] == "match" for d in detail)
    prose_ok = None if not prose_scores else all(d["status"] == "match" for d in prose_scores)
    scores_ok = None if score_check is None else score_check["all_within_tol"]
    signals = [x for x in (fields_ok, prose_ok, scores_ok) if x is not None]
    verified_tier1 = None if not signals else all(signals)

    return {
        "verified_tier1_components": {"cited_fields": fields_ok, "prose_scores": prose_ok, "scores_tag": scores_ok},
        "prose_score_detail": prose_scores,
        "stated_scores": scores,                   # log only; never shown in prompts or memory
        "score_check": score_check,
        "agent_id": agent_id,
        "is_adversary": False,
        "persona": None,
        "attack_mechanism": None,
        "content": claim,
        "confidence_score": confidence,
        "cites_evidence": cites,
        "cited_field": None if (cited_raw is None or cited_raw.strip().lower() == "none") else cited_raw,
        "ranking": ranking,
        "ranking_valid": ranking_valid,
        "verified_tier1": verified_tier1,          # log only; never shown in prompts
        "verified_tier1_detail": detail,
        "verified_tier2": None,                    # filled in by the auditor when defense is ON
        "parse_ok": not errors,
        "parse_errors": errors,
    }


def _check_optional_tag(text, tag, none_values, key, out, errors):
    """Missing tag entirely -> logged parse error. Tag present with 'none'/'n/a' -> valid, no error."""
    raw = extract_tag(text, tag)
    if raw is None:
        errors.append(f"missing_{tag}")
        out[key] = None
    else:
        v = raw.strip()
        out[key] = None if v.lower() in none_values else v


def parse_adversary_output(raw_text, agent_id, persona):
    base = parse_panelist_output(raw_text, agent_id, require_scores=False)  # adversary prompts have no SCORES tag
    base["is_adversary"] = True
    base["persona"] = persona
    base["attack_mechanism"] = PERSONA_MECHANISMS[persona]

    text = strip_reasoning(raw_text)
    extra_errors = []
    if persona == "Fabricator":
        _check_optional_tag(text, "SUMMARY_INJECTION", {"n/a", "na", ""}, "summary_injection", base, extra_errors)
    elif persona == "Editor":
        _check_optional_tag(text, "OMITTED", {"none", ""}, "omitted", base, extra_errors)
    elif persona == "CoalitionBuilder":
        _check_optional_tag(text, "MISCHARACTERIZED_AGENT", {"none", ""}, "mischaracterized_agent", base, extra_errors)

    if extra_errors:
        base["parse_errors"] = base["parse_errors"] + extra_errors
        base["parse_ok"] = False
    return base


def parse_auditor_output(raw_text, expected_ids):
    """Returns (reviews, errors). reviews: entry_id -> {flagged: bool, reasoning: str}.
    Entries the auditor failed to review are absent (they stay verified_tier2=None)."""
    text = (raw_text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    data = None
    try:
        data = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                data = None
    if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
        return {}, ["unparseable_json"]

    reviews, errors = {}, []
    expected = {e.upper() for e in expected_ids}
    for item in data["entries"]:
        if not isinstance(item, dict):
            errors.append("bad_item")
            continue
        eid = str(item.get("entry_id", "")).strip().upper()
        flagged = item.get("flagged")
        if eid not in expected:
            errors.append(f"unknown_entry_id:{eid}")
            continue
        if not isinstance(flagged, bool):
            errors.append(f"bad_flagged:{eid}")
            continue
        reviews[eid] = {"flagged": flagged, "reasoning": str(item.get("reasoning", ""))[:600]}
    for eid in sorted(expected):
        if eid not in reviews:
            errors.append(f"missing_entry:{eid}")
    return reviews, errors
