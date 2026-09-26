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

[ ] A Playwright spec outside the regular e2e suite walks the demo through the console and
    captures the README's screenshots and the frames of the GIF; CI does not run it
[ ] A `readme-media` script in the orchestrator runs that spec against the running stack,
    then writes optimized PNGs and an animated GIF into `docs/images/`, each under the
    repository's 500 KiB large-file limit
[ ] The only new dependency is Pillow, in the development group only (never in an image),
    named and justified in the report
[ ] The script's own logic has unittest tests; `selftest` is green

## 2. The README

[ ] Rewritten for a newcomer: what it is, what it looks like, how to run it, the
    five-minute demo, how it works, what is inside, status and scope, where to read more
[ ] Facts only: no capability described that does not exist; the deferred AI stage is
    named as deferred, once, not presented as a layer of the platform
[ ] An architecture diagram GitHub renders (Mermaid), the services in one table, the two
    data paths and the safety rules that make the PLC the only way to a valve
[ ] Screenshots of the main screens and the GIF of the demo, with alt text
[ ] Every link and anchor resolves; every command in it is one the orchestrator has

## 3. The 15-minute check

[ ] Reading time measured (words at a conservative reading speed) and kept under ten
    minutes, leaving the rest for the stack to come up
[ ] The README's instructions followed literally on this checkout, from `uv sync` to a
    signed-in console and a green `smoke`, and timed; the clean-clone timing of
    2026-09-19 quoted for a machine without caches
[ ] The README checked as GitHub renders it (Mermaid, images, GIF) from the pushed branch

## 4. Completion

[ ] State documents: the overview (the new script), the commands module and the command
    reference, the rule changelog, `AGENTS.md` inside its budget, the ROADMAP progress note
[ ] Full gate green; CI green on the pushed branch
[ ] Checklist filled honestly; report in Russian
[ ] Merged into `main` fast-forward, branch deleted locally and on `origin`; `main` not
    pushed without the owner's word
