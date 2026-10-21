"""aggregation.py -- deterministic Borda voting, herding diagnostics, and metrics against the reference."""
import config as cfg
from candidate_profiles_upgraded import CANDIDATE_LIST, REFERENCE_RANKING, REFERENCE_SCORES, REFERENCE_TOP_K


def _rate(a, b):
    return None if b == 0 else round(a / b, 4)


def vote_diagnostics(round_turns):
    """How many DISTINCT rankings were cast, and what share of later voters copied the first valid
    voter's ranking. distinct == 1 means the round's votes carried one vote's worth of information."""
    valid = [t["ranking"] for t in round_turns if t["ranking_valid"]]
    if not valid:
        return {"n_valid": 0, "n_distinct_rankings": 0, "first_speaker_agreement": None, "degenerate": False}
    first = valid[0]
    agreement = None if len(valid) < 2 else round(sum(1 for r in valid[1:] if r == first) / (len(valid) - 1), 4)
    distinct = len({tuple(r) for r in valid})
    return {"n_valid": len(valid), "n_distinct_rankings": distinct,
            "first_speaker_agreement": agreement, "degenerate": len(valid) >= 3 and distinct == 1}


def borda_aggregate(round_turns, weights=None):
    n = len(CANDIDATE_LIST)
    points = {cid: 0.0 for cid in CANDIDATE_LIST}
    valid = 0
    for t in round_turns:
        if not t["ranking_valid"]:
            continue
        valid += 1
        w = 1.0 if weights is None else weights.get(t["agent_id"], 1.0)
        for pos, cid in enumerate(t["ranking"]):
            points[cid] += w * (n - 1 - pos)
    points = {cid: round(p, 6) for cid, p in points.items()}
    ranking = sorted(CANDIDATE_LIST, key=lambda cid: (-points[cid], cid)) if valid else []
    tied = sorted(cid for cid in CANDIDATE_LIST if list(points.values()).count(points[cid]) > 1) if valid else []
    return {
        "ranking": ranking,
        "method": "borda_count" if weights is None else "weighted_borda_count",
        "borda_points": points,
        "vote_weights": weights,
        "n_valid_votes": valid,
        "n_votes": len(round_turns),
        "tied_candidates": tied,  # ties broken by candidate_id order
    }


def compare_to_reference(ranking):
    if not ranking:
        return None
    overlap = len(set(ranking[:cfg.TOP_K]) & set(REFERENCE_TOP_K))
    displacement = sum(abs(ranking.index(c) - REFERENCE_RANKING.index(c)) for c in CANDIDATE_LIST)
    regret = (sum(REFERENCE_SCORES[c] for c in REFERENCE_TOP_K)
              - sum(REFERENCE_SCORES[c] for c in ranking[:cfg.TOP_K]))
    return {
        "top_k_overlap": overlap,
        "top_k_error": cfg.TOP_K - overlap,
        "total_displacement": displacement,
        "decision_regret": round(regret, 4),
    }
