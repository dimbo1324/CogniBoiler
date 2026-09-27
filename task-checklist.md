# Task: fix known defect Д17 before 1.0 — the superheater at low flow and the restart

Owner decision of 2026-09-26 on Q8: fix Д17 before 1.0. The defect, from the roadmap: the
superheater has no metal heat capacity, so at low steam flow its outlet jumps (below
saturation on a sudden flow, to furnace gas temperature behind shut valves), and a restart
after an E-Stop reset at ten times real speed trips the unit again on high main steam
temperature. Investigation on 2026-09-26 added a fourth finding: with the PLC's commands
reaching the plant one step late, the unit trips even at a steady 250 MW — the inner flow
and temperature loops have no margin for delay.

Acceptance: the demo scenario of VISION §7 plays at 10× without the 3× restart workaround;
a test restarts the unit under a PLC command delay and passes; the nominal operating points
stay where they were; the full gate is green.

Publishing `main` was not asked for in this task.

Marks: `[ ]` open, `+` done, `-` not done or partially done (with a note).

**Closed without work on 2026-09-26:** the owner switched this session to a security and
code-quality audit before any code was written. Д17 stays open in the roadmap and the decision
log (Q8: fix before 1.0); this plan can be restored from commit `ada0fc4`.

## 1. Preparation

+ Q8 recorded as decided in the decision log
- The defect reproduced in lockstep with a command delay: restart and steady load —
  investigated (the fourth finding is in the decision log), no reproduction committed

## 2. Plant model (physics-engine)

- Superheater tube metal as a state with its heat capacity: gas heats the metal, the
    metal heats the steam; the outlet stays between saturation and the metal temperature
- Main steam temperature measured where the trip and the spray loop assume it — after
    the attemperator, before the governing valve
- Operating points and scenario initial states start with the metal in equilibrium;
    nominal readings stay put
- Tests for the new physics: energy balance, bounds, the time constant, no jump behind
    shut valves

## 3. Control (plc-controller)

- Inner loops (fuel flow, feedwater flow, steam temperature) retuned to hold with one
    and two steps of command delay, with the physical reasoning in the commit body
- A restart test under a command delay, and a steady-load test under delay

## 4. Workaround removed

- `demo` and the README recording restart at full demo speed again
- The live stack: `demo` at 10× plays green with clean logs

## 5. Completion

- State documents: Д17 closed in the roadmap, the known gap removed from the overview,
    the command reference and README without the 3× note
- Full gate green; CI on the pushed branch
- Checklist filled honestly; report in Russian
- Merged into `main` fast-forward, branch deleted locally and on `origin`; `main` not
    pushed without the owner's word
