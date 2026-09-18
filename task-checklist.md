# Task: two invariants and debt Д15 (fuel, emissions, part-load efficiency)

Owner instruction of 2026-09-18, answering the final report of the previous task: publish
`main`, adopt both proposed invariants, and take debt Д15. Work on branch
`fix/emissions-and-part-load-efficiency`, merged into `main` locally after a green gate and
published, since the owner asked for the push in this task.

Marks: `[ ]` open, `+` done, `-` not done or partially done (with a note).

## Publishing

+ `main` pushed to `origin` (408b33d..c2d59e2): S2–S4, S5, S8 and S10 are now on GitHub

## Invariants

+ I10 "the audit log is append-only" and I11 "every write from a protocol edge goes through
  the gateway" added to `docs/architecture/invariants.md`, each with how it is enforced
  today and the debt that still weakens it (Д13)
+ I9 states the fact after 2026-09-16: the AUTO hold test runs in lockstep
+ Decision recorded in the internal decision log

## Debt Д15 — the plant model

+ One fuel definition: pipeline natural gas with inerts at 42 MJ/kg; stoichiometric air
  14.44 kg/kg (0.344 kg per MJ) and CO2 2.343 kg/kg (55.8 g per MJ) derived from it
+ Flue gas Cp 1300 J/(kg·K) and furnace inventory 2115 kg chosen so that mass flow × Cp and
  mass × Cp stay at their calibrated values: the thermal design point does not move
+ NOx left as it was: the Zeldovich law is anchored on the flame-zone temperature and the
  plant feeds exactly that (furnace exit + 300 K, 1700 K at rated, ≈45 ppmv). The 2157 ppmv
  in the previous report came from my measurement script passing the adiabatic flame
  temperature; the stack check caught it
+ Governing valve throttles at part load (Stodola) and the isentropic efficiency takes a
  part-load penalty; net efficiency no longer rises at part load
+ Heat rate 9803 kJ/kWh at 300 MW, minimum 9781 at 275 MW, 9994 at 180 MW
- Sliding pressure, which would avoid the throttling loss, not implemented — it stays the
  development recorded under open question Q5

## Verification

+ Operating points 180–300 MW solve with zero derivatives; 8 h open loop without drift
+ Closed loop with the real PLC: hold 250 MW, 250→300, 180→300, feedwater pump drill (trip
  at 0.463 m, refused reset, reset, recovery), hot start, burner fouling, steam leak
+ 172 physics and PLC tests; full quality gate green (13/13)
+ Stack rebuilt with the new physics: containers healthy, smoke 13/13, and through the
  gateway 297 MW gives 37.2 % net, 9680 kJ/kWh, 540 kg CO2/MWh, 39.6 ppmv NOx and stack
  442.6 K, while 180 MW gives 36.35 % and 9905 kJ/kWh — the part-load penalty is visible
  on the running unit
+ One existing test adapted, none deleted or weakened: the inlet-pressure case compared two
  states a throttle-governed machine cannot reach; a new case pins the throttling

## Completion

+ State documents: ROADMAP (Д15 closed, S2 amendment), architecture overview, decision log,
  invariants
+ Checklist filled honestly
+ Final report in Russian; merged into `main` and pushed, as the owner asked
