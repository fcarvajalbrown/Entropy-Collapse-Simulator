# Extension plan: construction-journal follow-up ("paper 2")

Roadmap for a substantive extension of the Entropy Robustness Index work aimed
at a construction / structural-engineering journal (e.g. Revista de la
Construccion, which desk-rejected the current paper on discipline, novelty, and
general-significance grounds). See `manuscript/CLAUDE.md` for submission status.

Scope decision (2026-07-05): **hybrid, engine stays linear-elastic.** No
nonlinear or dynamic solver is added; every hard invariant in the repo-root
`CLAUDE.md` and `THEORY.md` is preserved. The lift is in framing, a realistic
case, and validation against code-style assessment, not in new physics.

## Why the current paper does not fit a construction journal

The three rejection words map to real gaps, none fixable by rewording:

- **Discipline.** It reads as a computational-mechanics method paper. A
  construction journal wants a structural-engineering contribution tied to how
  buildings are actually assessed.
- **Novelty.** `THEORY.md` already concedes entropy-of-strain-energy is not new
  on its own. The genuine novelty (the R_S formulation, the determinacy bound,
  non-reducibility to compliance, the verified implementation) is real but is
  under-sold as engineering value.
- **Significance.** Planar, linear-elastic, quasi-static, member-level. For a
  construction reader that caps practical weight; the honest limitations section
  currently hands the editor the rejection.

## The reframe that makes it a construction paper

Present R_S not as "an interesting metric" but as **a calibration-free screening
index for progressive-collapse vulnerability of steel frames.**

Value proposition a practicing engineer recognises: code robustness assessment
(GSA 2003; UFC 4-023-03, both already cited) requires choosing which columns to
notionally remove and running expensive nonlinear alternate-load-path (ALP)
analyses against structure-specific thresholds (drift limit, acceptance DCR).
R_S needs no threshold and ranks members by criticality (`dH_k`) from a single
linear solve. It is therefore cheap triage that tells the engineer **where to
spend the expensive nonlinear ALP effort.** That is a workflow contribution, not
a curiosity, and it is honest about the linear-elastic scope (a screening step,
not a replacement for nonlinear ALP).

## Work items

All four keep the engine linear-elastic and add to the existing modules and test
phases; keep all 10 test phases green and numbers reproducible from live runs.

1. **Realistic, recognisable case.**
   Current scenarios are toy-scale (`building_2d` is 2-bay 3-story). Add a
   published-benchmark steel building with provenance, ideally the Ziemian steel
   benchmark frames already cited in `THEORY.md`. Implement as a new
   `structure/frames/frame_<name>.py` exposing `build() -> FrameData` and
   register it in `simulation/scenarios.py`. The study must read as construction,
   not a demo.

2. **Show R_S triage agrees with code ALP.**
   Extend `analysis/importance.py`: does the entropy criticality ranking
   (`dH_k`, from `entropy/robustness.py`) pick the same critical columns that a
   GSA / UFC single-column-removal analysis flags (mechanism or gross
   overstress)? Report rank agreement (Spearman, as already done for
   compliance). Agreement substantiates the screening claim; a documented
   divergence is still a publishable, honest finding.

3. **Tie it to a design decision.**
   Two variants of the same building (baseline vs. one strengthened / added-
   redundancy bay). Show R_S distinguishes them and picks the more robust one,
   and that the entropy critical-member ranking points at the members whose
   strengthening most improves R_S. Significance becomes "this changes what you
   build." Natural home: a new analysis routine plus a manuscript figure.

4. **Reframe limitations as scope-of-a-screening-tool.**
   Rewrite Section 8 so the linear-elastic / planar / quasi-static limits read
   as the defined envelope of a triage index (flagging where nonlinear ALP is
   worth running), not as an apology. Keep the scientific-honesty invariant: no
   predictive-early-warning claim, entropy drop stays a coincident localization
   marker.

Optional heavier lift, deferred (not in this scope): a plastic-hinge
nonlinear-static ALP for one column-removal case, to show the cheap linear R_S
ranking survives under code-faithful nonlinear analysis. Most convincing to a
construction reviewer, but the biggest effort and it touches the
no-phenomenological-redistribution invariant. Revisit only if a reviewer asks.

## Strategic framing

- **This is a separate follow-up paper, not an RCUC resubmission.** It is
  substantial enough to be a distinct contribution (application plus validation
  against code ALP), which also dissolves the single-venue exclusivity block:
  paper 1 (the R_S formulation) stays at LAJSS; this is paper 2. If LAJSS
  rejects, paper 2 absorbs and replaces it.
- **Code and analysis work can start now; only submission waits on LAJSS.**
- **No honest rewrite guarantees acceptance.** The planar / linear-elastic
  nature is intrinsic; a construction editor may still find it insufficiently
  significant. This plan raises the odds substantially; it cannot promise a yes.

## Suggested sequence

1. Build and register the benchmark frame scenario (item 1); add a test phase
   case and validate against `benchmark.py`-style checks.
2. Implement the R_S-vs-code-ALP agreement study (item 2).
3. Implement the design-variant decision demo (item 3).
4. Draft the reframed manuscript around the screening-tool narrative (items 3, 4)
   in a new `manuscript/<JOURNAL>/` folder once the target journal is chosen,
   importing the shared `compute_results()` so numbers stay live.
5. Hold submission until LAJSS resolves.
