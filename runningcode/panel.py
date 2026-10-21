"""panel.py -- the 6-round panel loop: turns, memory, rotation, adversary entry/exit, auditor, transcript."""
import json
import os
import uuid
from datetime import datetime, timezone

import candidate_profiles_upgraded
import config as cfg
import llm
from aggregation import borda_aggregate, compare_to_reference, vote_diagnostics
from candidate_profiles_upgraded import (CANDIDATES, CANDIDATE_LIST, REFERENCE_RANKING, REFERENCE_SCORES, CUTOFF_MARGIN)
from conditions import (adversary_seats_for_condition, condition_label_for_count,
                        exit_round_actual, last_round_present)
from defense import compute_vote_weights, run_auditor, visible_memory
from fingerprint import config_fingerprint
from parsing import parse_adversary_output, parse_panelist_output
from prompts_adversary import (PERSONA_SYSTEM_PROMPTS, build_coalition_user_prompt,
                               build_editor_user_prompt, build_fabricator_user_prompt)
from prompts_benign import BENIGN_SYSTEM_PROMPT, build_panelist_user_prompt
from records import (append_summary_csv, checkpoint_path, load_checkpoint, save_checkpoint,
                     summarize_run)
from verification import verify_prose_scores


def run_condition(condition, run_seed, replication, temperature, peer_check, defense_on, pacer,
                  adversary_count=1):
    target, suppress = candidate_profiles_upgraded.TARGET_CANDIDATE, candidate_profiles_upgraded.SUPPRESS_CANDIDATE
    label = "baseline_no_adversary" if condition is None else condition_label_for_count(condition, adversary_count)
    ckpt_path = checkpoint_path(label, defense_on, replication)
    resume = load_checkpoint(ckpt_path, run_seed) if defense_on else None
    run_id = resume["run_id"] if resume else \
        f"{label}-def{'on' if defense_on else 'off'}-rep{replication}-s{run_seed}-{uuid.uuid4().hex[:6]}"
    print(f"=== {'Resuming' if resume else 'Starting'} run {run_id} ===")
    print(f"Model: {cfg.NVIDIA_MODEL} | Rounds: {cfg.TOTAL_ROUNDS} | Panel: {cfg.PANEL_SIZE} | "
          f"temp: {temperature} | seed base: {run_seed} | peer_check: {peer_check} | "
          f"defense: {'ON (auditor ' + cfg.AUDITOR_MODEL + ')' if defense_on else 'off'}")
    if condition is not None:
        print(f"Condition: {label} | target: {target} | suppress: {suppress}")
    print()

    seats = [f"A{i + 1}" for i in range(cfg.PANEL_SIZE)]
    adv_seat_indices = []
    adversary_active_seats = set()
    if condition is not None:
        adv_seat_indices = adversary_seats_for_condition(condition, adversary_count)
        if condition.entry_mode == "seeded":
            adversary_active_seats = set(adv_seat_indices)  # present from round 1 -- no entry event

    exit_round = exit_round_actual(condition)
    summary_round = last_round_present(condition)

    shared_memory, rotation_events, rounds_log = [], [], []
    next_agent_num = cfg.PANEL_SIZE + 1
    call_idx = 0
    models_reported = set()
    max_prompt_tokens = 0

    start_round = 1
    if resume:
        seats = resume["seats"]
        adv_seat_indices = resume.get("adversary_seats", adv_seat_indices)
        if "adversary_active_seats" in resume:
            adversary_active_seats = set(resume["adversary_active_seats"])
        elif resume.get("adversary_active"):
            # Checkpoints written before two-adversary support used one seat.
            adversary_active_seats = set(adv_seat_indices)
        shared_memory = resume["shared_memory"]
        rotation_events = resume["rotation_events"]
        rounds_log = resume["rounds_log"]
        next_agent_num = resume["next_agent_num"]
        call_idx = resume["call_idx"]
        models_reported = set(resume["models_reported"])
        max_prompt_tokens = resume["max_prompt_tokens"]
        start_round = resume["round_number"] if resume["phase"] == "pre_audit" else resume["round_number"] + 1
        print(f"  [resume] from {resume['phase']} checkpoint of round {resume['round_number']}; "
              f"continuing at round {start_round}")

    def _checkpoint(phase, round_number, turns):
        if not defense_on:
            return  # defense-off runs never write or read checkpoints
        save_checkpoint(ckpt_path, {
            "run_id": run_id, "run_seed": run_seed, "config_fingerprint": config_fingerprint(),
            "auditor_model": cfg.AUDITOR_MODEL, "phase": phase, "round_number": round_number,
            "seats": seats, "adversary_seats": adv_seat_indices,
            "adversary_active_seats": sorted(adversary_active_seats),
            "shared_memory": shared_memory,
            "rotation_events": rotation_events, "rounds_log": rounds_log,
            "next_agent_num": next_agent_num, "call_idx": call_idx,
            "models_reported": sorted(models_reported), "max_prompt_tokens": max_prompt_tokens,
            "current_round_turns": turns,
        })

    for round_number in range(start_round, cfg.TOTAL_ROUNDS + 1):
        print(f"--- Round {round_number} ---")

        # Injected entry fires before this round's turns.
        if (condition is not None and condition.entry_mode == "injected"
                and round_number == condition.entry_round and not adversary_active_seats):
            for adv_seat_idx in adv_seat_indices:
                old_agent = seats[adv_seat_idx]
                new_agent = f"A{next_agent_num}"
                next_agent_num += 1
                seats[adv_seat_idx] = new_agent
                adversary_active_seats.add(adv_seat_idx)
                rotation_events.append({
                    "round": round_number, "seat_idx": adv_seat_idx,
                    "agent_removed": old_agent, "agent_added": new_agent,
                    "reason": "adversary_entry", "coincides_with_scheduled_rotation": False,
                })
                print(f"  [adversary entry] seat {adv_seat_idx}: {old_agent} -> {new_agent} ({condition.persona})")

        # Memory the panelists will see this round. With defense ON, flagged content whose source
        # has left the panel is discounted here (spec 10.5).
        mem_view = visible_memory(shared_memory, set(seats), defense_on)
        discounted_ids = [m["entry_id"] for m in shared_memory
                          if m["round_introduced"] < round_number and m not in mem_view
                          ] if defense_on and cfg.DISCOUNT_MODE == "exclude" else \
                         [m["entry_id"] for m in shared_memory
                          if defense_on and m.get("verified_tier2") is False
                          and m["source_agent"] not in set(seats) and m["round_introduced"] < round_number]
        if discounted_ids:
            print(f"  [defense] discounting {len(discounted_ids)} flagged memory entr"
                  f"{'y' if len(discounted_ids) == 1 else 'ies'} from departed agents: {discounted_ids}")

        resume_pre_audit = (bool(resume) and resume["phase"] == "pre_audit"
                            and round_number == resume["round_number"])
        current_round_turns = resume["current_round_turns"] if resume_pre_audit else []
        for seat_idx, agent_id in ([] if resume_pre_audit else enumerate(seats)):
            is_adv_turn = condition is not None and seat_idx in adversary_active_seats
            persona = condition.persona if is_adv_turn else None

            if is_adv_turn:
                system_prompt = PERSONA_SYSTEM_PROMPTS[persona]
                if persona == "Fabricator":
                    is_summary_turn = round_number == summary_round
                    user_prompt = build_fabricator_user_prompt(
                        round_number, current_round_turns, mem_view, target, suppress, is_summary_turn)
                elif persona == "Editor":
                    user_prompt = build_editor_user_prompt(
                        round_number, current_round_turns, mem_view, target, suppress)
                else:  # CoalitionBuilder
                    user_prompt = build_coalition_user_prompt(
                        round_number, current_round_turns, mem_view, target)
            else:
                system_prompt = BENIGN_SYSTEM_PROMPT
                user_prompt = build_panelist_user_prompt(round_number, current_round_turns, mem_view, peer_check)

            call_idx += 1
            call_seed = run_seed + call_idx
            print(f"  {agent_id} thinking...", end=" ", flush=True)
            resp = llm.call_nvidia(system_prompt, user_prompt, pacer, seed=call_seed, temperature=temperature)

            turn = parse_adversary_output(resp["text"], agent_id, persona) if is_adv_turn \
                else parse_panelist_output(resp["text"], agent_id)
            turn.update({
                "raw_output": resp["text"],
                "usage": resp["usage"],
                "seed": call_seed,
                "model_reported": resp["model"],
                "system_fingerprint": resp["fingerprint"],
            })
            if resp["model"]:
                models_reported.add(resp["model"])
            max_prompt_tokens = max(max_prompt_tokens, resp["usage"].get("prompt_tokens", 0))
            current_round_turns.append(turn)

            snippet = turn["content"][:70] + ("..." if len(turn["content"]) > 70 else "")
            flag = "" if turn["parse_ok"] else \
                f"  [PARSE: {', '.join(e.split(':')[0] for e in turn['parse_errors'])}]"
            adv_flag = f"  [ADV:{persona}]" if is_adv_turn else ""
            print(f"-> \"{snippet}\"{flag}{adv_flag}")

            if turn["content"]:
                shared_memory.append({
                    "entry_id": f"M{len(shared_memory) + 1}",
                    "claim": turn["content"],
                    "source_agent": agent_id,
                    "round_introduced": round_number,
                    "entry_type": "claim",
                    "is_adversary": is_adv_turn,   # analysis metadata only; never shown in any prompt
                    "verified_tier1": turn["verified_tier1"],
                    "verified_tier1_detail": turn["verified_tier1_detail"],
                    "verified_tier2": None,  # filled by the auditor when defense is ON
                })

            # Shared-summary poisoning: a SEPARATE memory entry, same agent_id, no marker.
            if is_adv_turn and persona == "Fabricator" and turn.get("summary_injection"):
                shared_memory.append({
                    "entry_id": f"M{len(shared_memory) + 1}",
                    "claim": turn["summary_injection"],
                    "source_agent": agent_id,
                    "round_introduced": round_number,
                    "entry_type": "summary_injection",
                    "is_adversary": True,
                    "verified_tier1": (lambda d: None if not d else all(x["status"] == "match" for x in d))(
                        verify_prose_scores(turn["summary_injection"])),
                    "verified_tier1_detail": [],
                    "verified_tier2": None,
                })
            # Editor's OMITTED is logged in the turn dict only; never written to shared memory.

        # ---- Tier 2 + weighted aggregation (defense ON) or plain Borda (defense OFF) ----
        auditor_review = None
        if defense_on:
            _checkpoint("pre_audit", round_number, current_round_turns)
            round_entries = [m for m in shared_memory if m["round_introduced"] == round_number]
            prior_entries = [m for m in shared_memory if m["round_introduced"] < round_number]
            print("  auditor reviewing...", end=" ", flush=True)
            auditor_review = run_auditor(round_number, current_round_turns, round_entries, prior_entries)
            print(f"-> flagged agents: {auditor_review['flagged_agents'] or 'none'}"
                  f"{'' if auditor_review['parse_ok'] else '  [AUDITOR PARSE: ' + ', '.join(auditor_review['parse_errors'][:3]) + ']'}")
            weights, reasons = compute_vote_weights(current_round_turns, round_entries)
            agg_result = borda_aggregate(current_round_turns, weights)
            agg_result["weight_reasons"] = reasons
        else:
            agg_result = borda_aggregate(current_round_turns)
        agg_result["vs_reference"] = compare_to_reference(agg_result["ranking"])
        agg_result["diagnostics"] = vote_diagnostics(current_round_turns)
        dg = agg_result["diagnostics"]
        print(f"  {agg_result['method']} -> {agg_result['ranking']} "
              f"({agg_result['n_valid_votes']}/{agg_result['n_votes']} valid votes; "
              f"{dg['n_distinct_rankings']} distinct rankings{' -- DEGENERATE' if dg['degenerate'] else ''})")

        rounds_log.append({
            "round_number": round_number,
            "speaking_order": list(seats),
            "turns": current_round_turns,
            "memory_state": list(shared_memory),
            "memory_discounted_entry_ids": discounted_ids,
            "aggregation_result": agg_result,
            "auditor_review": auditor_review,  # None when defense is off
        })

        # Adversary exit -- may coincide with the regular schedule's slot for this round (spec 4.3).
        adversary_exited_this_round = set()
        if condition is not None and adversary_active_seats and exit_round == round_number:
            for adv_seat_idx in sorted(adversary_active_seats):
                coincident = (round_number in cfg.ROTATION_SCHEDULE
                              and cfg.ROTATION_SCHEDULE[round_number] == adv_seat_idx)
                old_agent = seats[adv_seat_idx]
                new_agent = f"A{next_agent_num}"
                next_agent_num += 1
                seats[adv_seat_idx] = new_agent
                adversary_exited_this_round.add(adv_seat_idx)
                rotation_events.append({
                    "round": round_number, "seat_idx": adv_seat_idx,
                    "agent_removed": old_agent, "agent_added": new_agent,
                    "reason": "adversary_exit", "coincides_with_scheduled_rotation": coincident,
                })
                print(f"  [adversary exit] seat {adv_seat_idx}: {old_agent} -> {new_agent}"
                      f"{' (coincides with scheduled rotation)' if coincident else ''}")
            adversary_active_seats.clear()

        if round_number in cfg.ROTATION_SCHEDULE:
            seat_idx = cfg.ROTATION_SCHEDULE[round_number]
            if seat_idx in adversary_exited_this_round:
                pass  # already logged as the single combined event above -- don't double it
            else:
                old_agent = seats[seat_idx]
                new_agent = f"A{next_agent_num}"
                next_agent_num += 1
                seats[seat_idx] = new_agent
                rotation_events.append({
                    "round": round_number, "agent_removed": old_agent, "agent_added": new_agent,
                    "reason": "scheduled", "coincides_with_scheduled_rotation": False,
                })
                print(f"  [rotation] seat {seat_idx}: {old_agent} -> {new_agent}")
        print()

    final_ranking = rounds_log[-1]["aggregation_result"]["ranking"]
    ground_truth_diff = [
        {"candidate_id": cid,
         "final_position": final_ranking.index(cid) + 1 if cid in final_ranking else None,
         "reference_position": REFERENCE_RANKING.index(cid) + 1,
         "reference_score": REFERENCE_SCORES[cid]}
        for cid in CANDIDATE_LIST
    ]

    if models_reported - {cfg.NVIDIA_MODEL}:
        print(f"WARNING: API reported model(s) {sorted(models_reported)} but requested {cfg.NVIDIA_MODEL}")

    model_assignment = {"panelist": cfg.NVIDIA_MODEL}
    if defense_on:
        model_assignment["auditor"] = cfg.AUDITOR_MODEL

    transcript = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "condition": {
            "entry_round": None if condition is None else condition.entry_round,
            "exit_round": None if condition is None else condition.exit_round,
            "persona": None if condition is None else condition.persona,
            "defense_state": "on" if defense_on else "off",
            "label": label,
            "replication": replication,
            "target_candidate": None if condition is None else target,
            "suppress_candidate": None if condition is None else suppress,
            "adversary_count": 0 if condition is None else adversary_count,
            "adversary_seat": adv_seat_indices[0] if adv_seat_indices else None,
            "adversary_seats": adv_seat_indices,
        },
        "config": {
            "prompt_version": cfg.PROMPT_VERSION,
            "panel_size": cfg.PANEL_SIZE,
            "rounds": cfg.TOTAL_ROUNDS,
            "top_k": cfg.TOP_K,
            "turnover_schedule": cfg.ROTATION_SCHEDULE,
            "model_assignment": model_assignment,
            "models_reported": sorted(models_reported),
            "temperature": temperature,
            "base_seed": run_seed,
            "peer_check": peer_check,
            "adversary_count": 0 if condition is None else adversary_count,
            "show_peer_rankings": cfg.SHOW_PEER_RANKINGS,
            "score_tol": cfg.SCORE_TOL,
            "config_fingerprint": config_fingerprint(),
            "memory_policy": "all_prior_rounds_full_no_window",
            "aggregation": "weighted_borda_count_in_code" if defense_on else "borda_count_in_code",
            "defense": {
                "downweight_factor": cfg.DOWNWEIGHT_FACTOR,
                "discount_mode": cfg.DISCOUNT_MODE,
                "auditor_temperature": cfg.AUDITOR_TEMPERATURE,
                "auditor_thinking_budget": cfg.AUDITOR_THINKING_BUDGET,
            } if defense_on else None,
            "reference_scores": REFERENCE_SCORES,
            "cutoff_margin": CUTOFF_MARGIN,
        },
        "candidates": CANDIDATES,
        "rounds": rounds_log,
        "rotation_events": rotation_events,
        "final_ranking": final_ranking,
        "reference_ranking": REFERENCE_RANKING,
        "ground_truth_diff": ground_truth_diff,
        "final_metrics": compare_to_reference(final_ranking),
    }

    out_path = os.path.join(cfg.OUTPUT_DIR, f"{run_id}_transcript.json")
    with open(out_path, "w") as f:
        json.dump(transcript, f, indent=2)
    if defense_on and os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    row = summarize_run(run_id, run_seed, replication, temperature, peer_check, defense_on,
                        condition, rounds_log, final_ranking, max_prompt_tokens, adversary_count)
    append_summary_csv(row)

    def pct(x):
        return "n/a" if x is None else f"{x:.1%}"

    print(f"Parse-ok turns: {pct(row['parse_ok_rate'])} of {row['n_turns']}")
    print(f"Tier 1: {pct(row['fact_error_rate'])} of {row['n_facts']} cited values wrong | "
          f"{pct(row['claim_error_rate'])} of {row['n_checked_claims']} checkable claims flagged")
    if defense_on:
        print(f"Tier 2: {row['n_entries_flagged']} entries flagged "
              f"({row['adv_entries_flagged']} adversary, {row['benign_entries_flagged']} benign) | "
              f"first adversary detection round: {row['first_adv_detection_round']} | "
              f"downweighted votes: {row['n_downweighted_votes']} | "
              f"auditor parse-ok: {pct(row['auditor_parse_ok_rate'])}")
    print(f"Tier 1 prose scores: {pct(row['prose_score_error_rate'])} of {row['n_prose_scores']} weighted scores "
          f"stated in claim text are wrong")
    print(f"Tier 1 derived scores: {pct(row['score_set_error_rate'])} of {row['n_score_sets']} stated score sets "
          f"off by > {cfg.SCORE_TOL} (mean abs err {row['score_mae_mean']}); ranking follows own scores in "
          f"{pct(row['ranking_follows_own_scores_rate'])}")
    print(f"Herding: mean distinct rankings/round {row['mean_distinct_rankings']} | first-speaker agreement "
          f"{pct(row['mean_first_speaker_agreement'])} | degenerate rounds {row['n_degenerate_rounds']}/{cfg.TOTAL_ROUNDS}")
    print(f"Max prompt tokens seen: {max_prompt_tokens} (NVIDIA RPM cap {cfg.NVIDIA_RPM_LIMIT}) | "
          f"NVIDIA tokens used this run: {row['total_nvidia_tokens']}")
    if condition is None and row["attack_success"] == 1:
        print(f"\n*** NO HEADROOM: this no-adversary baseline already places target {target} in the "
              f"top-{cfg.TOP_K}. Adversary runs will now be refused. Run --calibrate and set TARGET_OVERRIDE / "
              f"SUPPRESS_OVERRIDE in config.py (or fix the task) first. ***\n")
    if condition is None and row["n_degenerate_rounds"] >= cfg.TOTAL_ROUNDS - 1:
        print("*** WARNING: nearly every round had a single distinct ranking -- the panel is voting as one. ***\n")
    print(f"Final ranking:     {final_ranking}")
    print(f"Reference ranking: {REFERENCE_RANKING}")
    if condition is not None:
        print(f"Attack success (target {target} in final top-{cfg.TOP_K}): {bool(row['attack_success'])}")
    print(f"vs reference: {transcript['final_metrics']}")
    print(f"Transcript written to {out_path}\n")
    return out_path
