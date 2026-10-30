# Internal peer review of the LAJSS 9169 revision

Independent review of `revision_9169.md` and the seven figures, run before
resubmission with Felipe's peer-review template. Verdict: Major Revision.

Felipe's decision: fix the confirmed factual errors only (entropy-trigger
description, line 93 localization sentence, "load-independent" in line 21,
author affiliation) and resubmit; M1-M9 are left for a later round or paper 2.

Claims verified in code before acting: DROP_FLOOR = 1e-3 and min_history = 5 in
`analysis/criteria.py:170-197`; the line 93 sentence contradicts Table 2 (frame
worst case 0.390 < truss 0.737); line 21 says R_S is load-independent; R_S /
(1 - mechanism fraction) from Table 3 is 0.84, 0.83, 0.82, 0.81, 0.78, 0.79.

## Summary

R_S averages, over single notional removals, the normalized Shannon entropy of
the member strain-energy distribution in a first-order linear-elastic planar
model (H_k = 0 for mechanisms), on three benchmarks; it rises with DSI in a
counter-diagonal sweep, its ranking is not a monotone transform of compliance
importance, and the entropy criterion fires with the compliance criterion and
one step after DCR.

## Strengths

- Honest framing; no early-warning claim; calibrated novelty statement.
- Reproducible; sound three-tier verification design.
- Simple, bounded, well-defined index; equations and conventions match code.
- All four criteria compared on one incremental history.

## Major concerns

- M1. Redundancy tracking comes from the mechanism fraction: R_S = (1 - f_mech)
  x R_S,stable, and R_S,stable is flat (0.84 to 0.79) across DSI 1-6. Report
  R_S,stable (analysis/parametric.py:63), add baselines 1 - f_mech and the
  redundancy matrix (von Scheven et al., 2021), recast claim (i) if needed.
- M2. The moment-frame ranking gives ground-floor columns 0-2 the most negative
  dH_k; the largest-compliance-importance member has dH_k about -0.17. Compare
  with post-removal max DCR (analysis/importance.py:207 alp_column_severity,
  :232 compare_column_alp) and discuss whether higher H can mean a more
  dangerous state.
- M3. Entropy trigger as implemented uses DROP_FLOOR = 1e-3 nats when the
  baseline std is at most 1e-3 (always so before the first failure), min_history
  = 5, and the change in un-normalized S; the z = 3 branch never fires. Describe
  it as implemented, test sensitivity to DROP_FLOOR, reword "no threshold".
- M4. DCR criterion and removal rule are the same test (solver/failure.py:67,
  criteria.py:241), and metrics are recorded before removals, so the one-step
  lag is structural. Figure 6 has one post-failure point; the "recovers part of
  the way" sentence (line 135) is not shown. Report step-size and threshold
  studies (analysis/parametric.py:150, :78).
- M5. Loading and removal sets not representative of ALP practice: joint loads
  only, no distributed or lateral load; every member removed, not only columns;
  brittle removal at first yield with no buckling reduction.
- M6. Line 21 wrongly calls R_S load-independent; only scale invariance holds.
- M7. Discretization dependence: N counts elements; the single-span beam is two
  elements; define R_S over physical members or fix a meshing convention. The
  determinacy bound is proved only for pin-jointed trusses (THEORY.md 5.3).
- M8. Verification is internal (same author's second code); add an OpenSees
  check. Ensemble statistics overstated: report n, tie handling
  (analysis/importance.py:108-112), bootstrap or permutation intervals.
- M9. Missing robustness literature: Baker, Schubert and Faber (2008) Structural
  Safety 30(3):253-267; Starossek and Haberland (2011) Structure and
  Infrastructure Engineering 7(7-8):625-631; Biondini, Frangopol and Restelli
  (2008) ASCE Structures Congress; EN 1991-1-7. Put the determinacy proof in an
  appendix instead of "the accompanying theory note".

## Minor concerns

- Line 93: "sharper localization than anything in the moment frame" is wrong
  (frame worst case 0.390 vs truss 0.737).
- Line 21 claim (iii) "reaches the collapse onset with no calibration"
  conflicts with lines 23 and 154 and with M3.
- 0.05 m is a vertical deflection limit on the truss (about L/600), not a drift.
- "Pratt" truss: diagonals 19-24 all slope one way; a Pratt mirrors them about
  midspan. Say whether left-to-right counter-diagonal order affects monotonicity.
- Figure 6 x-axis is load step, Table 4 uses load factor; the "first member
  failure" line sits at step 26 while the failure is detected at step 25.
- Notation: Delta H_k vs dH_k in text and figure axes.
- Figure titles make claims; use neutral titles.
- Figure 2 member-id tick labels too small.
- Table 1 "Index checks 7/7" undefined; "0 means identical" next to 1e-14 %
  mixes conventions.
- Repository paths in Table 2-4 captions belong in Data Availability.
- No Discussion section; limitations sit under Results; caveat repeated.
- Informal register in line 154 ("I want to be precise", "The honest claim").
- Inconsistent heading capitalization.
- Shannon (1948) listed but not cited; cite at Equation (2).
- Feng et al. (2024) reference incomplete.
- Ziemian and Ziemian (2021) cited only "for future checks".
- Line 59 bending-plane bug is implementation detail.
- Report the stability eigenvalue tolerance 1e-9 (entropy/robustness.py:175).
- Archive the code with a DOI (Zenodo release).
- Title says "Steel Frames" though one benchmark is a truss.

## Questions

1. R_S over physical members or finite elements; effect of subdividing members?
2. R_S,stable and worst-case H_k per DSI in Table 3?
3. Why does removing a ground-floor column raise H; does the ranking agree with
   post-removal max DCR?
4. Does the trigger ever fire through the z-score branch; sensitivity to
   DROP_FLOOR 1e-4 and 1e-2?
5. Why does Figure 6 end one step after the first failure; where is the
   partial recovery?

## Note outside the review

The author block read "Instituto Virginio Gómez, Chillán, Chile"; Felipe is not
affiliated with it and the required line is "M.Sc. in Numerical Simulation in
Engineering (Universidad Politécnica de Madrid, Spain), Independent Researcher,
Santiago (Chile)".
