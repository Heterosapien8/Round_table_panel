"""
run.py -- the ONLY file you run.  One execution = one run, condition set in config.py.

Setup (PowerShell):
    $env:GROQ_API_KEY="your-groq-key"
    $env:GEMINI_API_KEY="your-gemini-key"     # only needed when DEFENSE_ON = True

    python run.py --selftest      # offline checks, no API calls
    python run.py --calibrate     # ~5 cheap blind calls: is there headroom? (re-runs reuse saved ballots; --fresh re-samples)
    python run.py                 # exactly one run, then exits

Edit settings in config.py, prompts in prompts_*.py, candidates in candidates.py.
Baseline gate: adversary runs are refused until --calibrate has passed for the current
prompts/candidates/target and no baseline run of the same config already puts the target in the top-k.
"""
import sys

import candidates
import config as cfg
from calibration import baseline_gate_check, calibrate, print_dataset_sanity
from conditions import condition_label, resolve_run_config
from llm import AuditorUnavailableError, DailyLimitError, TPMPacer
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
        condition = resolve_run_config(cfg.PERSONA_FABRICATOR, cfg.PERSONA_EDITOR, cfg.PERSONA_COALITION_BUILDER,
                                       cfg.ENTRY_EXIT, cfg.REPLICATION, cfg.PEER_CHECK)
    except ValueError as e:
        print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)

    if not cfg.GROQ_API_KEY:
        print("Error: GROQ_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    if cfg.DEFENSE_ON and not cfg.GEMINI_API_KEY:
        print("Error: DEFENSE_ON = True needs the GEMINI_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)
    if cfg.GROQ_MODEL != cfg.SPEC_MODEL:
        print(f"WARNING: GROQ_MODEL={cfg.GROQ_MODEL} differs from spec model {cfg.SPEC_MODEL}")

    label = "baseline_no_adversary" if condition is None else condition_label(condition)
    prior = existing_transcripts(label, cfg.DEFENSE_ON, cfg.REPLICATION)
    if prior and not cfg.ALLOW_RERUN:
        print(f"A transcript already exists for {label} / defense {'on' if cfg.DEFENSE_ON else 'off'} / "
              f"replication {cfg.REPLICATION}:\n  {prior[0]}\n"
              "Change REPLICATION or the condition, or set ALLOW_RERUN = True to repeat it.", file=sys.stderr)
        sys.exit(1)

    gate_ok, gate_msgs = baseline_gate_check(condition)
    for msg in gate_msgs:
        print(("BLOCKED: " if not gate_ok else "") + msg, file=sys.stderr if not gate_ok else sys.stdout)
    if not gate_ok:
        print("(Set ENFORCE_BASELINE_GATE = False in config.py to override -- results from an adversary run "
              "on an invalid baseline are not interpretable.)", file=sys.stderr)
        sys.exit(1)

    run_seed = cfg.BASE_SEED + (cfg.REPLICATION - 1) * 1000
    print_dataset_sanity()
    print()
    print(f"Configured run: {label} | defense {'ON' if cfg.DEFENSE_ON else 'off'} | "
          f"replication {cfg.REPLICATION} | seed {run_seed}\n")
    pacer = TPMPacer(cfg.TPM_LIMIT)
    try:
        run_condition(condition, run_seed, cfg.REPLICATION, cfg.TEMPERATURE, cfg.PEER_CHECK, cfg.DEFENSE_ON, pacer)
    except DailyLimitError as e:
        print(f"\nSTOPPED: {e}\nThis run was not saved. Rerun the same config once the quota resets.",
              file=sys.stderr)
        sys.exit(2)
    except AuditorUnavailableError as e:
        print(f"\nSTOPPED: {e}\nProgress is checkpointed. Rerun the same config to resume at the auditor step.",
              file=sys.stderr)
        sys.exit(2)
    except RuntimeError as e:
        print(f"\nRun failed: {e}\nThis run was not saved.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()