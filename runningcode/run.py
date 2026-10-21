"""
run.py -- the ONLY file you run. Matrix mode runs all configured cases for one replication.

Setup (PowerShell):
    $env:NVIDIA_API_KEY="your-nvidia-key"
    $env:GEMINI_API_KEY="your-gemini-key"     # only needed when DEFENSE_ON = True

    python run.py --selftest      # offline checks, no API calls
    python run.py --calibrate     # ~5 cheap blind calls: is there headroom? (re-runs reuse saved ballots; --fresh re-samples)
    python run.py                 # matrix mode: 8 cases; 16 when MATRIX_INCLUDE_DEFENSE = True

Edit settings in config.py, prompts in prompts_*.py, and candidate profiles in candidate_profiles_upgraded.py.
Baseline gate: adversary runs are refused until --calibrate has passed for the current
prompts/candidates/target and no baseline run of the same config already puts the target in the top-k.
"""
import sys

import candidate_profiles_upgraded
import config as cfg
from calibration import baseline_gate_check, calibrate, print_dataset_sanity
from conditions import (ENTRY_EXIT_KEYS, condition_label_for_count, resolve_run_config,
                        selected_persona, validate_adversary_count)
from llm import AuditorUnavailableError, DailyLimitError, NIMRatePacer
from panel import run_condition
from records import existing_transcripts


def main():
    if "--selftest" in sys.argv:
        from selftest import selftest
        selftest()
        return
    if "--calibrate" in sys.argv:
        calibrate()
        return

    try:
        persona = selected_persona(cfg.PERSONA_FABRICATOR, cfg.PERSONA_EDITOR,
                                   cfg.PERSONA_COALITION_BUILDER)
        if cfg.RUN_CASE_MATRIX and persona is not None:
            counts = tuple(cfg.ADVERSARY_COUNTS)
            if len(counts) != 2 or set(counts) != {1, 2}:
                raise ValueError("ADVERSARY_COUNTS must contain exactly (1, 2) in matrix mode.")
            for count in counts:
                validate_adversary_count(count)
            defense_states = (False, True) if cfg.MATRIX_INCLUDE_DEFENSE else (cfg.DEFENSE_ON,)
            cases = [
                (resolve_run_config(cfg.PERSONA_FABRICATOR, cfg.PERSONA_EDITOR,
                                    cfg.PERSONA_COALITION_BUILDER, entry_exit,
                                    cfg.REPLICATION, cfg.PEER_CHECK, count), count, defense_on)
                for entry_exit in ENTRY_EXIT_KEYS
                for count in counts
                for defense_on in defense_states
            ]
        else:
            count = 0 if persona is None else cfg.ADVERSARY_COUNT
            condition = resolve_run_config(
                cfg.PERSONA_FABRICATOR, cfg.PERSONA_EDITOR, cfg.PERSONA_COALITION_BUILDER,
                cfg.ENTRY_EXIT, cfg.REPLICATION, cfg.PEER_CHECK,
                1 if persona is None else count)
            cases = [(condition, count, cfg.DEFENSE_ON)]
    except (TypeError, ValueError) as e:
        print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)

    if not cfg.NVIDIA_API_KEY:
        print("Error: NVIDIA_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    if any(defense_on for _, _, defense_on in cases) and not cfg.GEMINI_API_KEY:
        print("Error: DEFENSE_ON = True needs the GEMINI_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)
    if cfg.NVIDIA_MODEL != cfg.SPEC_MODEL:
        print(f"WARNING: NVIDIA_MODEL={cfg.NVIDIA_MODEL} differs from spec model {cfg.SPEC_MODEL}")

    run_seed = cfg.BASE_SEED + (cfg.REPLICATION - 1) * 1000
    print_dataset_sanity()
    print()
    defense_summary = ("off/on" if cfg.RUN_CASE_MATRIX and cfg.MATRIX_INCLUDE_DEFENSE
                       else "ON" if cases[0][2] else "off")
    print(f"Configured execution: {len(cases)} case(s) | persona {persona or 'baseline'} | "
          f"defense {defense_summary} | peer check {cfg.PEER_CHECK} | "
          f"replication {cfg.REPLICATION} | seed {run_seed}\n")

    pending = []
    for condition, adversary_count, defense_on in cases:
        label = "baseline_no_adversary" if condition is None else condition_label_for_count(condition, adversary_count)
        prior = existing_transcripts(label, defense_on, cfg.REPLICATION)
        if prior and not cfg.ALLOW_RERUN:
            if len(cases) == 1:
                print(f"A transcript already exists for {label} / defense {'on' if defense_on else 'off'} / "
                      f"replication {cfg.REPLICATION}:\n  {prior[0]}\n"
                      "Change REPLICATION or the condition, or set ALLOW_RERUN = True to repeat it.",
                      file=sys.stderr)
                sys.exit(1)
            print(f"Skipping completed case {label} (replication {cfg.REPLICATION}).")
            continue
        pending.append((condition, adversary_count, defense_on, label))

    if not pending:
        print("All configured cases already have transcripts; nothing to run.")
        return

    gate_checked = False
    pacer = NIMRatePacer(cfg.NVIDIA_RPM_LIMIT)
    for condition, adversary_count, defense_on, label in pending:
        if condition is not None and not gate_checked:
            gate_ok, gate_msgs = baseline_gate_check(condition)
            for msg in gate_msgs:
                print(("BLOCKED: " if not gate_ok else "") + msg,
                      file=sys.stderr if not gate_ok else sys.stdout)
            if not gate_ok:
                print("(Set ENFORCE_BASELINE_GATE = False in config.py to override -- results from an adversary run "
                      "on an invalid baseline are not interpretable.)", file=sys.stderr)
                sys.exit(1)
            gate_checked = True

        print(f"\n### Case {label} / defense {'ON' if defense_on else 'off'} "
              f"({pending.index((condition, adversary_count, defense_on, label)) + 1}/{len(pending)}) ###")
        try:
            run_condition(condition, run_seed, cfg.REPLICATION, cfg.TEMPERATURE, cfg.PEER_CHECK,
                          defense_on, pacer, adversary_count)
        except DailyLimitError as e:
            print(f"\nSTOPPED in {label}: {e}\nRerun the same config once the quota resets.", file=sys.stderr)
            sys.exit(2)
        except AuditorUnavailableError as e:
            print(f"\nSTOPPED in {label}: {e}\nProgress is checkpointed. Rerun the same config to resume at the auditor step.",
                  file=sys.stderr)
            sys.exit(2)
        except RuntimeError as e:
            print(f"\nRun failed in {label}: {e}\nThis case was not saved.", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
