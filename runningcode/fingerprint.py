"""fingerprint.py -- hashes that decide whether a calibration/baseline 'counts' for the current config."""
import hashlib
import json

import candidates
import config as cfg
from prompts_benign import BENIGN_SYSTEM_PROMPT, build_panelist_user_prompt


def config_fingerprint():
    """Hash of everything that changes what the benign panel is asked to do or what counts as the
    attacker's goal."""
    material = json.dumps({
        "prompt": build_panelist_user_prompt(1, [], [], cfg.PEER_CHECK),
        "system": BENIGN_SYSTEM_PROMPT,
        "prompt_version": cfg.PROMPT_VERSION, "model": cfg.GROQ_MODEL, "top_k": cfg.TOP_K,
        "target": candidates.TARGET_CANDIDATE, "suppress": candidates.SUPPRESS_CANDIDATE,
        "show_peer_rankings": cfg.SHOW_PEER_RANKINGS,
        "temperature": cfg.TEMPERATURE, "peer_check": cfg.PEER_CHECK, "max_tokens": cfg.MAX_TOKENS_PER_TURN,
    }, sort_keys=True)
    return hashlib.sha1(material.encode()).hexdigest()[:10]


def ballot_input_fingerprint():
    """Everything that determines what the blind calibration ballots ARE. The attacker goal is
    deliberately NOT in here: changing it cannot change a ballot, so saved ballots are re-judged for free."""
    material = json.dumps({
        "prompt": build_panelist_user_prompt(1, [], [], cfg.PEER_CHECK), "system": BENIGN_SYSTEM_PROMPT,
        "model": cfg.GROQ_MODEL, "temperature": cfg.TEMPERATURE, "seed_base": cfg.CALIB_SEED_BASE,
        "n": cfg.CALIBRATION_BALLOTS, "max_tokens": cfg.MAX_TOKENS_PER_TURN,
    }, sort_keys=True)
    return hashlib.sha1(material.encode()).hexdigest()[:10]