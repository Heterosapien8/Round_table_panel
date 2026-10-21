"""conditions.py -- entry/exit conditions, persona resolution, and adversary seat logic."""
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
    return condition_label_for_count(c, 1)


def condition_label_for_count(c, adversary_count=1):
    """Return a stable output label, distinguishing one- and two-adversary cases."""
    if adversary_count not in (1, 2):
        raise ValueError("adversary_count must be 1 or 2")
    exit_label = "noexit" if c.exit_round == 6 else f"exit{c.exit_round}"
    base = f"{c.entry_mode}_r{c.entry_round}_{exit_label}__{c.persona.lower()}"
    return base if adversary_count == 1 else f"{base}__adv{adversary_count}"


CONDITION_BY_LABEL = {condition_label(c): c for c in CONDITIONS}
assert len(CONDITION_BY_LABEL) == 12, "condition labels collided"

ENTRY_EXIT_KEYS = ["seeded_r1_exit3", "seeded_r1_noexit", "injected_r3_exit4", "injected_r3_noexit"]


def selected_persona(fab, edi, coa):
    """Resolve the three persona flags to one persona name, or None for baseline."""
    chosen = [name for name, on in (("Fabricator", fab), ("Editor", edi), ("CoalitionBuilder", coa)) if on]
    if len(chosen) > 1:
        raise ValueError(f"Set at most ONE persona flag to True (you set: {', '.join(chosen)}).")
    return chosen[0] if chosen else None


def validate_adversary_count(adversary_count):
    if not isinstance(adversary_count, int) or adversary_count not in (1, 2):
        raise ValueError("ADVERSARY_COUNT values must be 1 or 2.")


def resolve_run_config(fab, edi, coa, entry_exit, replication, peer_check, adversary_count=1):
    """Turn the config flags into a Condition (or None for baseline). Raises ValueError on bad config."""
    from prompts_benign import PEER_CHECK_LINES
    chosen = selected_persona(fab, edi, coa)
    if peer_check not in PEER_CHECK_LINES:
        raise ValueError("PEER_CHECK must be 'off' or 'on'.")
    if not isinstance(replication, int) or replication < 1:
        raise ValueError("REPLICATION must be an integer >= 1.")
    validate_adversary_count(adversary_count)
    if not chosen:
        return None
    if entry_exit not in ENTRY_EXIT_KEYS:
        raise ValueError(f"ENTRY_EXIT must be one of {ENTRY_EXIT_KEYS}.")
    return CONDITION_BY_LABEL[f"{entry_exit}__{chosen.lower()}"]


def adversary_seats_for_condition(condition, adversary_count=1):
    """Choose seats that remain stable under the regular rotation schedule.

    For the injected/exit-round-4 condition, seat 1 intentionally coincides
    with the scheduled round-4 rotation. A second adversary uses seat 2 and
    exits independently. All other cases use seats 2 and 3, neither of which
    is touched by the regular schedule.
    """
    validate_adversary_count(adversary_count)
    if condition.entry_mode == "injected" and condition.exit_round == 4:
        return [1] if adversary_count == 1 else [1, 2]  # seat 1 == ROTATION_SCHEDULE[4]
    return [2] if adversary_count == 1 else [2, 3]


def adversary_seat_for_condition(condition):
    """Backward-compatible single-adversary seat helper."""
    return adversary_seats_for_condition(condition, 1)[0]


def exit_round_actual(condition):
    if condition is None:
        return None
    return condition.exit_round if condition.exit_round in (3, 4) else None  # 6 == no removal


def last_round_present(condition):
    if condition is None:
        return None
    er = exit_round_actual(condition)
    return er if er is not None else cfg.TOTAL_ROUNDS
