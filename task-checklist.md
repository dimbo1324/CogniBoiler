# Task: S13 step 6 — a README a stranger can run and understand in 15 minutes

Owner instruction of 2026-09-26: finish step 6 of the S13 plan to the very end. Step 6 is
the part of S13 that ROADMAP and VISION §10 describe as "README with screenshots and a GIF,
the current architecture, launch instructions", checked against the 1.0 criterion "the
README lets a stranger run and understand the project in 15 minutes".

Step 7 (the final VISION §10 tick-off, the v1.0.0 tag and release) is not part of this
task. Publishing `main` was not asked for in this task: the branch is merged locally and
the report says so.

Marks: `[ ]` open, `+` done, `-` not done or partially done (with a note).

## 1. Screenshots and a GIF that can be made again

+ A Playwright spec outside the regular e2e suite (`apps/web/readme/`, its own
  `playwright.readme.config.ts`) walks the demo through the console and captures the
  README's seven screenshots and the frames of the GIF; neither `console-e2e` nor CI runs it
+ A `readme-media` script in the orchestrator runs that spec against the running stack,
  then writes indexed-colour PNGs (21–57 KiB) and an animated GIF (382 KiB, 800 px,
  64 colours, 36 frames, 19 s) into `docs/images/`, each under the 500 KiB limit
+ The only new dependency is Pillow 12.3, in the development group only (never in an
  image), justified in the report
+ The script's own logic has 19 unittest tests; `selftest` is green

## 2. The README

+ Rewritten for a newcomer: what it is, what it looks like, how to run it, the
  five-minute demo with who does what and where to look, how it works, what is inside,
  quality and security, status and scope, development, documentation
+ Facts only: every statement checked against the code; the deferred AI stage is named as
  deferred, once, and no longer presented as a layer of the platform
+ An architecture diagram GitHub renders — validated with Mermaid 11 and redrawn top-down
  after the first layout proved unreadable at GitHub's width — the services in one table,
  the two data paths and the safety rules that make the PLC the only way to a valve
+ The demo GIF, the trip, and six screens in a gallery, each with alt text
+ Every link and anchor resolves and every command is in the orchestrator's catalogue
  (checked by script)

## 3. The 15-minute check

+ Reading time: 1722 words — 8.6 minutes at 200 words a minute, 11.5 at 150; the stack
  builds meanwhile
+ The README's instructions followed literally on this checkout: `uv sync` → `dev-secrets`
  → `stack up` → `smoke` in 128 s with warm caches, every command green; 4 min 34 s from an
  empty Docker on 2026-09-19. A second clean-clone run was not made: it would have
  created Docker volumes this agent is not allowed to delete
+ The README checked as GitHub renders it from the pushed branch: all 13 images load,
  all eleven sections render, the Mermaid block is present (its layout checked locally)

## 4. Found on the way — known defect Д17

+ Recording the GIF tripped the unit a second time after the E-Stop reset (672 °C main
  steam against the 580 °C limit). Traced to the superheater model at low steam flow and
  reproduced in lockstep with the PLC a step late; a fuel runback was tried, did not hold
  under scan lag, and was reverted rather than shipped
- Not fixed: it is plant-model and control work (S2/S3), and whether before 1.0 is the
  owner's decision — Q8 in the decision log, Д17 in the roadmap, a known gap in the
  overview
+ Worked around: `demo` restarts the unit at three times speed (plays green in 3 min 35 s,
  clean logs), the recording does the same and resets a latched trip before reloading a
  scenario; `test_plc_restart.py` pins the behaviour that holds today

## 5. Completion

+ State documents: the overview (the new script, the known gap), the commands module and
  the command reference, the rule changelog, `AGENTS.md` inside its budget (7 bytes to
  spare), the ROADMAP progress note and Д17, the decision log (Q8)
+ Full gate green (1085 pytest, 212 Vitest); CI on the pushed branch — see the report
+ Checklist filled honestly; report in Russian
+ Merged into `main` fast-forward, branch deleted locally and on `origin` — right after
  this commit, which cannot record it; `main` not pushed without the owner's word
