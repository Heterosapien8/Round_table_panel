"""verification.py -- Tier 1: deterministic, zero-API-cost checks against the ground-truth profiles."""
import re

import config as cfg
from candidates import CANDIDATE_BY_ID, CANDIDATE_LIST, REFERENCE_SCORES

NUM_RE = r"-?\d+(?:\.\d+)?|-?\.\d+"


def _norm_text(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", str(s).lower())).strip()


def verify_fact(fact):
    """Does the claimed raw value equal the profile value?"""
    cand = CANDIDATE_BY_ID.get(fact["candidate_id"])
    actual = None
    if cand is None:
        status = "unknown_candidate"
    elif fact["field"] not in cand["structured_evidence"]:
        status = "unknown_field"
    else:
        actual = cand["structured_evidence"][fact["field"]]
        if isinstance(actual, (int, float)):
            nums = re.findall(NUM_RE, fact["claimed_value"])
            ok = bool(nums) and abs(float(nums[0]) - float(actual)) < 1e-6
        else:
            ok = _norm_text(fact["claimed_value"]) == _norm_text(actual)
        status = "match" if ok else "mismatch"
    return {**fact, "actual_value": actual, "status": status}


def verify_scores(scores, ranking, ranking_valid):
    """Check the panelist's STATED derived scores against the reference scores. Also records whether
    the submitted ranking follows the panelist's own scores."""
    per, errs = {}, []
    for cid in CANDIDATE_LIST:
        err = abs(scores[cid] - REFERENCE_SCORES[cid])
        errs.append(err)
        per[cid] = {"stated": scores[cid], "actual": REFERENCE_SCORES[cid], "err": round(err, 4)}
    own_order = sorted(CANDIDATE_LIST, key=lambda c: (-scores[c], c))
    return {
        "per_candidate": per,
        "mae": round(sum(errs) / len(errs), 4),
        "max_err": round(max(errs), 4),
        "all_within_tol": all(e <= cfg.SCORE_TOL for e in errs),
        "ranking_matches_own_scores": bool(ranking_valid and own_order == ranking),
    }


_PROSE_NUM = r"(?<![\d.])(0?\.\d{2,4}|1\.0{1,4})(?!\d)"
# Tight forms: "C5 (0.922)", "C2 at 0.833", "C1: 0.744", "C3 scored 0.70", "C4 total of 0.66"
_PROSE_TIGHT = re.compile(
    r"\b(C\d+)\b\s*(?:\(|:|=|\bat\s|\bscores?\b(?:\s+of)?\s|\bscored\s|\btotal(?:s|ing)?\b(?:\s+of)?\s)\s*\(?\s*"
    + _PROSE_NUM)
# Loose form: "C5 ranks first with a total of 0.919" (no other candidate id or digit in between)
_PROSE_LOOSE = re.compile(
    r"\b(C\d+)\b[^\dC;()]{0,40}?\b(?:total|scores?|scored|at)\b(?:\s+of)?\s*\(?\s*" + _PROSE_NUM)
_LIST_TAIL = re.compile(r"\s*(?:,\s*)?(?:and\s+)?(?:0?\.\d{2,4})\b")


def extract_prose_scores(text):
    """Pull 'candidate id -> stated weighted score' pairs out of free-text claim prose.
    Conservative: only forms where the number clearly belongs to the preceding id."""
    found, seen = [], set()
    for rx, loose in ((_PROSE_TIGHT, False), (_PROSE_LOOSE, True)):
        for m in rx.finditer(text or ""):
            if loose:
                tail = text[m.end(): m.end() + 28]
                if _LIST_TAIL.match(tail) or "respectively" in tail.lower():
                    continue  # ambiguous list form, skip
            key = (m.group(1).upper(), float(m.group(2)), m.start(1))
            ident = (key[0], key[1])
            if ident in seen:
                continue
            seen.add(ident)
            found.append({"candidate_id": key[0], "claimed_score": key[1]})
    return found


def verify_prose_scores(text):
    """A weighted score STATED IN PROSE is checked against the reference score."""
    out = []
    for f in extract_prose_scores(text):
        actual = REFERENCE_SCORES.get(f["candidate_id"])
        if actual is None:
            continue
        err = abs(f["claimed_score"] - actual)
        out.append({**f, "actual_score": actual, "err": round(err, 4),
                    "status": "match" if err <= cfg.SCORE_TOL else "mismatch"})
    return out