"""defense.py -- Tier 2 auditor call, citation-weighted votes (10.4), provenance-aware discounting (10.5)."""
import config as cfg
import llm
from parsing import parse_auditor_output
from prompts_auditor import AUDITOR_SYSTEM_PROMPT, build_auditor_prompt


def run_auditor(round_number, turns, round_entries, prior_entries):
    """One batch auditor call (plus at most one retry if the reply is unparseable).
    Mutates round_entries[*]['verified_tier2'] and returns the spec-6.2 auditor_review block."""
    expected_ids = [m["entry_id"] for m in round_entries]
    prompt = build_auditor_prompt(round_number, turns, round_entries, prior_entries)
    reviews, errors, attempts, resp = {}, ["not_run"], 0, {}
    while attempts < cfg.AUDITOR_MAX_ATTEMPTS:
        attempts += 1
        resp = llm.call_gemini(AUDITOR_SYSTEM_PROMPT, prompt)
        reviews, errors = parse_auditor_output(resp["text"], expected_ids)
        if reviews:  # got something usable; missing entries are logged, not retried
            break

    for m in round_entries:
        r = reviews.get(m["entry_id"].upper())
        m["verified_tier2"] = None if r is None else (not r["flagged"])
        m["auditor_reasoning"] = None if r is None else r["reasoning"]

    flagged_entries = [m for m in round_entries if m["verified_tier2"] is False]
    return {
        "flagged_agents": sorted({m["source_agent"] for m in flagged_entries}),
        "flagged_claims": [
            {"entry_id": m["entry_id"], "agent_id": m["source_agent"], "claim": m["claim"],
             "reasoning": m["auditor_reasoning"]} for m in flagged_entries
        ],
        "reasoning": [{"entry_id": eid, **r} for eid, r in sorted(reviews.items())],
        "parse_ok": not errors,
        "parse_errors": errors,
        "attempts": attempts,
        "model_reported": resp.get("model"),
        "finish_reason": resp.get("finish_reason"),
        "usage": resp.get("usage"),
        "raw_output": resp.get("text"),
    }


def agent_tier2_status(entries):
    """False if any of the agent's entries this round was flagged; True if all were reviewed and
    cleared; None if the auditor produced no verdict."""
    vals = [e["verified_tier2"] for e in entries]
    if any(v is False for v in vals):
        return False
    if vals and all(v is True for v in vals):
        return True
    return None


def vote_weight(tier1, tier2):
    """Spec 10.4. Auditor flag or failed Tier 1 -> downweight. Otherwise full weight if the claim
    passed Tier 1 or was cleared by the auditor. Uncited AND not cleared -> downweight."""
    if tier2 is False or tier1 is False:
        return cfg.DOWNWEIGHT_FACTOR, "flagged_or_tier1_fail"
    if tier1 is True or tier2 is True:
        return 1.0, "verified"
    return cfg.DOWNWEIGHT_FACTOR, "uncited_unreviewed"


def compute_vote_weights(turns, round_entries):
    weights, reasons = {}, {}
    for t in turns:
        ents = [e for e in round_entries if e["source_agent"] == t["agent_id"]]
        t2 = agent_tier2_status(ents)
        t["verified_tier2"] = t2
        weights[t["agent_id"]], reasons[t["agent_id"]] = vote_weight(t["verified_tier1"], t2)
    return weights, reasons


def visible_memory(memory, panel_agent_ids, defense_on):
    """Spec 10.5: discount ONLY auditor-flagged content whose source agent has left the panel.
    Unflagged content from departed agents (and everything when defense is off) is untouched."""
    if not defense_on:
        return memory
    out = []
    for m in memory:
        if m.get("verified_tier2") is False and m["source_agent"] not in panel_agent_ids:
            if cfg.DISCOUNT_MODE == "exclude":
                continue
            out.append({**m, "claim": "[flagged by auditor as unsupported] " + m["claim"]})
        else:
            out.append(m)
    return out