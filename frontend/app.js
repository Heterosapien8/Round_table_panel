/**
 * ROTATING-MEMBERSHIP AI DECISION PANEL - FRONTEND APPLICATION SCRIPT
 * 
 * Strict constraints:
 * - Standalone execution (no external backend dependency)
 * - Formal and academic presentation
 * - Full support for loading any transcript JSON conforming to the benchmark schema
 */

(function () {
  'use strict';

  // Global State
  let transcriptData = null;
  let activeFileName = "baseline-bcfc6bb1_transcript.json";
  let activeRoundNumber = 1;

  // DOM Elements
  const fileInput = document.getElementById("fileInput");
  const uploadBtn = document.getElementById("uploadBtn");
  const loadDefaultBtn = document.getElementById("loadDefaultBtn");
  const dropZone = document.getElementById("dropZone");
  const browseLink = document.getElementById("browseLink");
  const copyJsonBtn = document.getElementById("copyJsonBtn");

  // Tab Navigation Elements
  const tabButtons = document.querySelectorAll(".tab-button");
  const tabPanes = document.querySelectorAll(".tab-pane");

  // Memory Filter Elements
  const memoryRoundFilter = document.getElementById("memoryRoundFilter");
  const memoryAgentFilter = document.getElementById("memoryAgentFilter");
  const memoryVerificationFilter = document.getElementById("memoryVerificationFilter");
  const memorySearchInput = document.getElementById("memorySearchInput");

  // Roman numeral converter
  function toRoman(num) {
    const lookup = { 1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII", 9: "IX", 10: "X" };
    return lookup[num] || num.toString();
  }

  // Initialize Application
  function init() {
    setupEventListeners();

    // Check if embedded default transcript is available
    if (window.DEFAULT_TRANSCRIPT && typeof window.DEFAULT_TRANSCRIPT === "object") {
      loadTranscriptData(window.DEFAULT_TRANSCRIPT, "baseline-bcfc6bb1_transcript.json");
    } else {
      showError("No default transcript embedded. Please upload a JSON transcript file.");
    }
  }

  // Event Listeners
  function setupEventListeners() {
    // Upload button
    uploadBtn.addEventListener("click", () => fileInput.click());
    browseLink.addEventListener("click", () => fileInput.click());

    // File input change
    fileInput.addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (file) {
        readFile(file);
      }
    });

    // Reset button
    loadDefaultBtn.addEventListener("click", () => {
      if (window.DEFAULT_TRANSCRIPT) {
        loadTranscriptData(window.DEFAULT_TRANSCRIPT, "baseline-bcfc6bb1_transcript.json");
      }
    });

    // Drag & Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
      dropZone.addEventListener(eventName, preventDefaults, false);
      document.body.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
      dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      const file = dt.files[0];
      if (file) {
        readFile(file);
      }
    });

    // Tab Switching
    tabButtons.forEach(btn => {
      btn.addEventListener("click", () => {
        tabButtons.forEach(b => b.classList.remove("active"));
        tabPanes.forEach(p => p.classList.remove("active"));
        
        btn.classList.add("active");
        const tabId = btn.getAttribute("data-tab");
        const targetPane = document.getElementById(tabId);
        if (targetPane) {
          targetPane.classList.add("active");
        }
      });
    });

    // Memory Filters
    memoryRoundFilter.addEventListener("change", renderMemoryTable);
    memoryAgentFilter.addEventListener("change", renderMemoryTable);
    memoryVerificationFilter.addEventListener("change", renderMemoryTable);
    memorySearchInput.addEventListener("input", renderMemoryTable);

    // Copy JSON
    copyJsonBtn.addEventListener("click", () => {
      if (!transcriptData) return;
      navigator.clipboard.writeText(JSON.stringify(transcriptData, null, 2))
        .then(() => {
          const origText = copyJsonBtn.textContent;
          copyJsonBtn.textContent = "JSON Copied Successfully";
          setTimeout(() => {
            copyJsonBtn.textContent = origText;
          }, 2000);
        })
        .catch(err => {
          alert("Failed to copy JSON: " + err);
        });
    });
  }

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  // Read Local File
  function readFile(file) {
    if (!file.name.toLowerCase().endsWith(".json")) {
      alert("Error: File must have a .json extension.");
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const parsed = JSON.parse(e.target.result);
        if (!validateTranscriptSchema(parsed)) {
          alert("Error: The uploaded JSON does not match the decision panel transcript schema.");
          return;
        }
        loadTranscriptData(parsed, file.name);
      } catch (err) {
        alert("Error parsing JSON file: " + err.message);
      }
    };
    reader.readAsText(file);
  }

  // Schema Validator
  function validateTranscriptSchema(data) {
    return data && 
           typeof data === 'object' &&
           Array.isArray(data.candidates) &&
           Array.isArray(data.rounds);
  }

  // Load Transcript Data & Render View
  function loadTranscriptData(data, fileName) {
    transcriptData = data;
    activeFileName = fileName || "transcript.json";
    activeRoundNumber = 1;

    renderStatusBar();
    renderMetadataCards();
    renderDeliberationTab();
    renderProgressionTab();
    renderCandidatesTab();
    populateMemoryFilters();
    renderMemoryTable();
    renderRotationsTab();
    renderRawJson();
  }

  // Render Status Bar
  function renderStatusBar() {
    document.getElementById("currentFileName").textContent = activeFileName;
    document.getElementById("currentRunId").textContent = transcriptData.run_id || "N/A";
    document.getElementById("currentTimestamp").textContent = transcriptData.timestamp ? new Date(transcriptData.timestamp).toUTCString() : "N/A";
  }

  // Render Top Metadata Cards
  function renderMetadataCards() {
    const cond = transcriptData.condition || {};
    const cfg = transcriptData.config || {};
    const finalR = transcriptData.final_ranking || [];
    const refR = transcriptData.reference_ranking || [];

    // Card I: Experimental Condition
    document.getElementById("metaLabel").textContent = cond.label || "baseline_no_adversary";
    document.getElementById("metaDefense").textContent = (cond.defense_state || "off").toUpperCase();
    document.getElementById("metaPersona").textContent = cond.persona ? cond.persona : "None (Benign Panel)";
    
    let timingText = "N/A (No Adversary)";
    if (cond.entry_round !== null && cond.entry_round !== undefined) {
      timingText = `Enters R${cond.entry_round} -> Exits R${cond.exit_round || 6}`;
    }
    document.getElementById("metaTiming").textContent = timingText;

    // Card II: Configuration
    document.getElementById("metaPanelSize").textContent = `${cfg.panel_size || 5} Seats`;
    document.getElementById("metaRounds").textContent = `${cfg.rounds || (transcriptData.rounds ? transcriptData.rounds.length : 6)} Rounds`;
    
    const models = cfg.model_assignment || {};
    document.getElementById("metaModelPanelist").textContent = models.panelist || "llama-3.1-8b-instant";
    document.getElementById("metaModelAggregator").textContent = models.aggregator || "llama-3.1-8b-instant";

    // Card III: Decision Outcome & Alignment
    document.getElementById("finalRankingDisplay").textContent = `[${finalR.join(", ")}]`;
    document.getElementById("refRankingDisplay").textContent = `[${refR.join(", ")}]`;

    const topMatch = finalR.length > 0 && refR.length > 0 && finalR[0] === refR[0];
    const exactMatch = JSON.stringify(finalR) === JSON.stringify(refR);
    let dispText = "Exact Ground Truth Match";
    if (!exactMatch && topMatch) {
      dispText = "Rank-1 Preserved (Sub-tier Displacement)";
    } else if (!topMatch) {
      dispText = "Rank-1 Displaced (Top Selection Error)";
    }
    document.getElementById("metaDisplacement").textContent = dispText;

    // Card IV: Verification Metrics
    let checkableCount = 0;
    let falseCount = 0;

    if (Array.isArray(transcriptData.rounds)) {
      transcriptData.rounds.forEach(r => {
        if (Array.isArray(r.turns)) {
          r.turns.forEach(t => {
            if (t.verified_tier1 !== null && t.verified_tier1 !== undefined) {
              checkableCount++;
              if (t.verified_tier1 === false) {
                falseCount++;
              }
            }
          });
        }
      });
    }

    document.getElementById("metaTotalClaims").textContent = checkableCount;
    const rate = checkableCount > 0 ? (falseCount / checkableCount * 100).toFixed(1) : 0;
    document.getElementById("metaFalseClaims").textContent = `${falseCount} (${rate}%)`;
    document.getElementById("metaNoiseFloor").textContent = `Active Sample (n=1)`;
    document.getElementById("metaTier2Status").textContent = (cond.defense_state === "on") ? "Defense ON (Audited)" : "Defense OFF (Disabled)";
  }

  // TAB 1: Render Deliberation Tab
  function renderDeliberationTab() {
    const rounds = transcriptData.rounds || [];
    const container = document.getElementById("roundPillsContainer");
    container.innerHTML = "";

    rounds.forEach(r => {
      const btn = document.createElement("button");
      btn.className = `round-pill ${r.round_number === activeRoundNumber ? "active" : ""}`;
      btn.type = "button";
      btn.textContent = `Round ${toRoman(r.round_number)}`;
      btn.addEventListener("click", () => {
        activeRoundNumber = r.round_number;
        renderDeliberationTab();
      });
      container.appendChild(btn);
    });

    const activeRound = rounds.find(r => r.round_number === activeRoundNumber) || rounds[0];
    if (!activeRound) return;

    // Title & Info
    document.getElementById("currentRoundTitle").textContent = `Round ${toRoman(activeRound.round_number)} Deliberation Details`;
    document.getElementById("roundSpeakingOrder").textContent = (activeRound.speaking_order || []).join(", ");

    // Check if rotation occurred in this round
    const rotations = transcriptData.rotation_events || [];
    const thisRoundRot = rotations.filter(ev => ev.round === activeRound.round_number);
    if (thisRoundRot.length > 0) {
      const desc = thisRoundRot.map(ev => `Seat turnover: ${ev.agent_removed} removed -> ${ev.agent_added} added (${ev.reason})`).join("; ");
      document.getElementById("roundRotationNote").textContent = desc;
    } else {
      document.getElementById("roundRotationNote").textContent = "None";
    }

    // Turns
    const turnsContainer = document.getElementById("turnsContainer");
    turnsContainer.innerHTML = "";

    (activeRound.turns || []).forEach((t, idx) => {
      const turnCard = document.createElement("div");
      turnCard.className = "turn-card";

      // Badges
      let roleBadge = t.is_adversary ? `[ADVERSARY: ${(t.persona || 'UNKNOWN').toUpperCase()}]` : `[BENIGN PANEL]`;
      let tier1Badge = `[TIER 1: N/A]`;
      if (t.verified_tier1 === true) {
        tier1Badge = `[TIER 1: VERIFIED TRUE]`;
      } else if (t.verified_tier1 === false) {
        tier1Badge = `[TIER 1: FLAGGED FALSE]`;
      }

      turnCard.innerHTML = `
        <div class="turn-header">
          <div class="turn-agent">Seat ${idx + 1} &mdash; Agent ${t.agent_id}</div>
          <div class="turn-badges">
            <span class="turn-badge">${roleBadge}</span>
            <span class="turn-badge">${tier1Badge}</span>
          </div>
        </div>
        <div class="turn-claim">${escapeHtml(t.content)}</div>
        <div class="turn-meta">
          <span><strong>Confidence Score:</strong> ${t.confidence_score !== null && t.confidence_score !== undefined ? t.confidence_score : 'N/A'}</span>
          <span><strong>Cites Evidence:</strong> ${t.cites_evidence ? 'True' : 'False'}</span>
          <span><strong>Cited Field(s):</strong> ${t.cited_field ? escapeHtml(t.cited_field) : 'None'}</span>
          ${t.attack_mechanism ? `<span><strong>Attack Mechanism:</strong> ${escapeHtml(t.attack_mechanism)}</span>` : ''}
        </div>
      `;
      turnsContainer.appendChild(turnCard);
    });

    // Aggregator
    const agg = activeRound.aggregation_result || {};
    document.getElementById("aggMethod").textContent = agg.method || "majority_vote";
    document.getElementById("aggRanking").textContent = `[${(agg.ranking || []).join(", ")}]`;
    document.getElementById("aggAuditor").textContent = activeRound.auditor_review ? JSON.stringify(activeRound.auditor_review) : "Defense Off (No auditor review)";
    document.getElementById("aggTallyRaw").textContent = agg.tally_raw || "(No raw tally logged)";
  }

  // TAB 2: Render Progression Tab
  function renderProgressionTab() {
    const rounds = transcriptData.rounds || [];
    const rotations = transcriptData.rotation_events || [];
    const tbody = document.getElementById("progressionTableBody");
    tbody.innerHTML = "";

    rounds.forEach(r => {
      const agg = r.aggregation_result || {};
      const ranking = agg.ranking || [];
      const topCand = ranking.length > 0 ? ranking[0] : "N/A";
      
      const rotMatch = rotations.find(ev => ev.round === r.round_number);
      let rotText = "&mdash;";
      if (rotMatch) {
        rotText = `${rotMatch.agent_removed} &rarr; ${rotMatch.agent_added} (${rotMatch.reason})`;
      }

      const row = document.createElement("tr");
      row.innerHTML = `
        <td><strong>Round ${toRoman(r.round_number)}</strong></td>
        <td>${(r.speaking_order || []).join(", ")}</td>
        <td>[${ranking.join(", ")}]</td>
        <td><strong>${topCand}</strong></td>
        <td>${rotText}</td>
      `;
      tbody.appendChild(row);
    });

    // Ground Truth Analysis Table
    const gtBody = document.getElementById("groundTruthTableBody");
    gtBody.innerHTML = "";
    const diffs = transcriptData.ground_truth_diff || [];

    if (diffs.length > 0) {
      diffs.forEach(d => {
        const delta = (d.final_position !== null && d.reference_position !== null) ? d.final_position - d.reference_position : "N/A";
        let assessment = "Preserved";
        if (delta > 0) assessment = `Deprioritized by +${delta} ranks`;
        if (delta < 0) assessment = `Boosted by ${delta} ranks`;

        const row = document.createElement("tr");
        row.innerHTML = `
          <td><strong>${d.candidate_id}</strong></td>
          <td>Rank ${d.final_position !== null ? d.final_position : 'N/A'}</td>
          <td>Rank ${d.reference_position !== null ? d.reference_position : 'N/A'}</td>
          <td>${delta === 0 ? '0 (Exact)' : (delta > 0 ? `+${delta}` : `${delta}`)}</td>
          <td>${assessment}</td>
        `;
        gtBody.appendChild(row);
      });
    } else {
      const cands = transcriptData.candidates || [];
      const finalR = transcriptData.final_ranking || [];
      const refR = transcriptData.reference_ranking || [];

      cands.forEach(c => {
        const cid = c.candidate_id;
        const finalPos = finalR.indexOf(cid) + 1;
        const refPos = refR.indexOf(cid) + 1;
        const delta = finalPos - refPos;
        let assessment = "Preserved";
        if (delta > 0) assessment = `Deprioritized by +${delta} ranks`;
        if (delta < 0) assessment = `Boosted by ${delta} ranks`;

        const row = document.createElement("tr");
        row.innerHTML = `
          <td><strong>${cid}</strong></td>
          <td>Rank ${finalPos > 0 ? finalPos : 'N/A'}</td>
          <td>Rank ${refPos > 0 ? refPos : 'N/A'}</td>
          <td>${delta === 0 ? '0 (Exact)' : (delta > 0 ? `+${delta}` : `${delta}`)}</td>
          <td>${assessment}</td>
        `;
        gtBody.appendChild(row);
      });
    }
  }

  // TAB 3: Render Candidates Tab
  function renderCandidatesTab() {
    const cands = transcriptData.candidates || [];
    const tbody = document.getElementById("candidatesTableBody");
    tbody.innerHTML = "";

    cands.forEach(c => {
      const ev = c.structured_evidence || {};
      const row = document.createElement("tr");
      row.innerHTML = `
        <td><strong>${c.candidate_id}</strong> &mdash; ${c.name_placeholder || c.candidate_id}</td>
        <td>${ev.technical_assessment_score !== undefined ? ev.technical_assessment_score : 'N/A'} / 100</td>
        <td>${ev.years_experience !== undefined ? ev.years_experience + ' Years' : 'N/A'}</td>
        <td>${ev.reference_score !== undefined ? ev.reference_score + ' / 10.0' : 'N/A'}</td>
        <td>${ev.relevant_projects_count !== undefined ? ev.relevant_projects_count : 'N/A'}</td>
        <td>${ev.highest_degree !== undefined ? ev.highest_degree : 'N/A'}</td>
        <td><strong>Rank ${c.reference_rank !== undefined ? c.reference_rank : 'N/A'}</strong></td>
      `;
      tbody.appendChild(row);
    });
  }

  // TAB 4: Shared Memory Filters & Render
  function populateMemoryFilters() {
    const rounds = transcriptData.rounds || [];
    
    // Populate rounds dropdown
    memoryRoundFilter.innerHTML = '<option value="all">All Rounds</option>';
    rounds.forEach(r => {
      const opt = document.createElement("option");
      opt.value = r.round_number;
      opt.textContent = `Round ${toRoman(r.round_number)}`;
      memoryRoundFilter.appendChild(opt);
    });

    // Collect all unique agents
    const agents = new Set();
    rounds.forEach(r => {
      (r.memory_state || []).forEach(m => {
        if (m.source_agent) agents.add(m.source_agent);
      });
    });

    memoryAgentFilter.innerHTML = '<option value="all">All Agents</option>';
    Array.from(agents).sort().forEach(a => {
      const opt = document.createElement("option");
      opt.value = a;
      opt.textContent = `Agent ${a}`;
      memoryAgentFilter.appendChild(opt);
    });
  }

  function renderMemoryTable() {
    const rounds = transcriptData.rounds || [];
    const tbody = document.getElementById("memoryTableBody");
    tbody.innerHTML = "";

    // The final round contains the complete cumulative shared memory ledger
    const lastRound = rounds[rounds.length - 1];
    const memoryList = lastRound ? (lastRound.memory_state || []) : [];

    const selectedRound = memoryRoundFilter.value;
    const selectedAgent = memoryAgentFilter.value;
    const selectedVerification = memoryVerificationFilter.value;
    const searchVal = memorySearchInput.value.trim().toLowerCase();

    let count = 0;
    memoryList.forEach(m => {
      // Filters
      if (selectedRound !== "all" && m.round_introduced.toString() !== selectedRound) {
        return;
      }
      if (selectedAgent !== "all" && m.source_agent !== selectedAgent) {
        return;
      }
      if (selectedVerification === "true" && m.verified_tier1 !== true) {
        return;
      }
      if (selectedVerification === "false" && m.verified_tier1 !== false) {
        return;
      }
      if (selectedVerification === "none" && (m.verified_tier1 === true || m.verified_tier1 === false)) {
        return;
      }
      if (searchVal && !m.claim.toLowerCase().includes(searchVal)) {
        return;
      }

      count++;
      let tier1Text = "N/A";
      if (m.verified_tier1 === true) tier1Text = "Verified (True)";
      if (m.verified_tier1 === false) tier1Text = "Flagged (False)";

      let tier2Text = m.verified_tier2 !== null && m.verified_tier2 !== undefined ? JSON.stringify(m.verified_tier2) : "Defense Off";

      const row = document.createElement("tr");
      row.innerHTML = `
        <td>Round ${toRoman(m.round_introduced)}</td>
        <td><strong>Agent ${m.source_agent}</strong></td>
        <td>${escapeHtml(m.claim)}</td>
        <td>${tier1Text}</td>
        <td>${tier2Text}</td>
      `;
      tbody.appendChild(row);
    });

    if (count === 0) {
      tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; font-style: italic; padding: 16px;">No shared memory records match the selected filter criteria.</td></tr>`;
    }
  }

  // TAB 5: Render Turnover & Rotations
  function renderRotationsTab() {
    const rotations = transcriptData.rotation_events || [];
    const tbody = document.getElementById("rotationsTableBody");
    tbody.innerHTML = "";

    if (rotations.length === 0) {
      tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; font-style: italic; padding: 16px;">No rotation events recorded in this experiment run.</td></tr>`;
      return;
    }

    const rounds = transcriptData.rounds || [];

    rotations.forEach(ev => {
      const targetRound = rounds.find(r => r.round_number === ev.round + 1) || rounds.find(r => r.round_number === ev.round);
      const speakingOrder = targetRound ? (targetRound.speaking_order || []).join(", ") : "N/A";

      const row = document.createElement("tr");
      row.innerHTML = `
        <td><strong>After Round ${toRoman(ev.round)}</strong></td>
        <td>Agent ${ev.agent_removed}</td>
        <td>Agent ${ev.agent_added}</td>
        <td>${ev.reason || 'scheduled'}</td>
        <td>${speakingOrder}</td>
      `;
      tbody.appendChild(row);
    });
  }

  // TAB 6: Render Raw JSON
  function renderRawJson() {
    document.getElementById("rawJsonOutput").textContent = JSON.stringify(transcriptData, null, 2);
  }

  // Helper: HTML Escaper
  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Initialize on load
  document.addEventListener("DOMContentLoaded", init);
})();
