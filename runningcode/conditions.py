"""conditions.py -- the 12 experimental conditions (4 entry/exit combos x 3 personas) and seat logic."""
from collections import namedtuple

import config as cfg

Condition = namedtuple("Condition", ["entry_mode", "entry_round", "exit_round", "persona"])

ENTRY_EXIT_COMBOS = [
    ("seeded", 1, 3),    # present rounds 1-3, 3-round observation window
    ("seeded", 1, 6),    # no removal within the run at all
    ("injected", 3, 4),  # present rounds 3-4, 2-round observation window
    ("injected", 3, 6),  # no removal within the run at all
]
PERSONAS = ["Fabricator", "Editor", "CoalitionBuilder"]

CONDITIONS = [
    Condition(entry_mode, entry_round, exit_round, persona)
    for (entry_mode, entry_round, exit_round) in ENTRY_EXIT_COMBOS
    for persona in PERSONAS
]
assert len(CONDITIONS) == 12


def condition_label(c):
    exit_label = "noexit" if c.exit_round == 6 else f"exit{c.exit_round}"
    return f"{c.entry_mode}_r{c.entry_round}_{exit_label}__{c.persona.lower()}"


CONDITION_BY_LABEL = {condition_label(c): c for c in CONDITIONS}
assert len(CONDITION_BY_LABEL) == 12, "condition labels collided"

ENTRY_EXIT_KEYS = ["seeded_r1_exit3", "seeded_r1_noexit", "injected_r3_exit4", "injected_r3_noexit"]


def resolve_run_config(fab, edi, coa, entry_exit, replication, peer_check):
    """Turn the config flags into a Condition (or None for baseline). Raises ValueError on bad config."""
    from prompts_benign import PEER_CHECK_LINES
    chosen = [name for name, on in (("Fabricator", fab), ("Editor", edi), ("CoalitionBuilder", coa)) if on]
    if len(chosen) > 1:
        raise ValueError(f"Set at most ONE persona flag to True (you set: {', '.join(chosen)}).")
    if peer_check not in PEER_CHECK_LINES:
        raise ValueError("PEER_CHECK must be 'off' or 'on'.")
    if not isinstance(replication, int) or replication < 1:
        raise ValueError("REPLICATION must be an integer >= 1.")
    if not chosen:
        return None
    if entry_exit not in ENTRY_EXIT_KEYS:
        raise ValueError(f"ENTRY_EXIT must be one of {ENTRY_EXIT_KEYS}.")
    return CONDITION_BY_LABEL[f"{entry_exit}__{chosen[0].lower()}"]


def adversary_seat_for_condition(condition):
    """Seat index chosen so entry/exit mechanics overlay the fixed non-adversary rotation schedule
    (cfg.ROTATION_SCHEDULE = {2: 4, 4: 1}) the way spec 4.3 requires: the injected/exit-round-4
    adversary exit coincides with the regular round-4 slot (seat 1) as ONE combined event; every
    other condition's adversary sits in seat 2, which the schedule never touches."""
    if condition.entry_mode == "injected" and condition.exit_round == 4:
        return 1  # == ROTATION_SCHEDULE[4], by design
    return 2      # untouched by ROTATION_SCHEDULE at any round


def exit_round_actual(condition):
    if condition is None:
        return None
    return condition.exit_round if condition.exit_round in (3, 4) else None  # 6 == no removal


def last_round_present(condition):
    if condition is None:
        return None
    er = exit_round_actual(condition)
    return er if er is not None else cfg.TOTAL_ROUNDS