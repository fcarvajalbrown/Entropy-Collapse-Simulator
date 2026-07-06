# Manuscript registry

Guidance for the manuscript deliverables and a live record of where the paper
has been submitted. If anything here conflicts with the repo-root `CLAUDE.md`,
`README.md`, or `THEORY.md`, those win.

## The paper

One paper, retargeted to several journals: *An Entropy Robustness Index for
Planar Steel Frames: Formulation, Verified Implementation, and Comparison with
Standard Collapse Criteria*. The research (the Entropy Robustness Index `R_S`
and its verified implementation) is identical across every target; only the
formatting, reference style, and front matter change per journal.

## Layout

Each journal has its own folder `manuscript/<JOURNAL>/` holding that journal's
build script, template, and built Word manuscript. Shared assets live at the
`manuscript/` root:

- `results.py` — the single `compute_results()`; every build script imports it
  so reported numbers come from the live analysis and cannot drift.
- `generate_figures.py`, `figures/` — figures regenerated from live runs.
- `manuscript.md` — plain reading copy.

Every builder does `from results import compute_results` with `manuscript/` on
`sys.path` via `MANU = HERE / ".."`. To add a journal, mirror the pattern in a
new `manuscript/<JOURNAL>/build_docx_<journal>.py`: import `compute_results`,
open the journal template as the base, clear the placeholder body, and refill
it with the `has_style` / `P` helpers, resolving paths relative to `HERE`.

## Submission status

Last updated: 2026-07-05.

Exclusivity: every journal submission affirms the paper is not under review
elsewhere. It is the same paper each time, so only ONE journal may hold it at a
time. Submit sequentially, never in parallel.

| Journal | Folder | Status (2026-07-05) |
|---|---|---|
| LAJSS (Latin American Journal of Solids and Structures) | `LAJSS/` | LIVE - passed initial screen (plagiarism, scope, formatting, English); assigned to peer reviewers. |
| Ingenieria e Investigacion (UNAL, Colombia) | `IngInv/` | Submission 128239 acknowledged; exact status unconfirmed - verify it is closed/withdrawn so it does not overlap LAJSS. |
| Applied and Computational Mechanics (Univ. West Bohemia, Czech) | `ACM/` | DESK-REJECTED - editorial priority screen, no peer review, no actionable critique. Dead; build kept for reference. |
| Revista de la Construccion / RCUC (PUC Chile) | `RDLC/` | DESK-REJECTED - discipline / novelty / general-significance screen; explicitly "not necessarily a reflection of the quality of your research." No peer review. Dead. |
| Ingeniare (Chile) | `Ingeniare/` | Staged in repo, not submitted. |

Notes on the two rejections: both were editorial desk-rejects on venue fit
(priority; discipline / novelty / significance), not peer review. Neither
returned reviewer comments, so there is nothing substantive to revise off them
before the next submission.

## Paper 2 venue plan (construction-journal extension)

Decided 2026-07-05. Paper 2 (the R_S screening-tool extension; see
`EXTENSION_PLAN.md`) targets **Revista de la Construccion (RDLC)** and commits to
the construction-practice reframe: R_S as a calibration-free progressive-collapse
screening/triage index for steel frames, anchored to the GSA / UFC 4-023-03
alternate-load-path (notional column removal) method, demonstrated on a realistic
steel building (the Vogel 1985 frame) with a **quantified triage payoff**. Match
the register of the one computational steel method paper RDLC has accepted. RDLC
is materials-science-dominated with no entropy/robustness-index precedent, so this
is an uphill fit (see the `rdlc-acceptance-bar` memory for the full acceptance-bar
research and the desk-screen checklist).

**Contingency (per Felipe, 2026-07-05):** if RDLC desk-rejects paper 2 (i.e. a
second RDLC rejection), do NOT resubmit to RDLC. Pivot to a better-fit structural
/ computational-mechanics venue (survey candidates such as Journal of
Constructional Steel Research, Structures, Engineering Structures, Journal of
Building Engineering, Ingeniare; LAJSS as fallback) at that point. Submission of
paper 2 stays blocked until LAJSS resolves paper 1.

**Paper 2 draft (2026-07-05):** drafted in `manuscript/RDLC_paper2/`
(`build_docx_rdlc2.py` -> `Carvajal_RDLC_screening_manuscript.docx`). It imports
`compute_paper2_results()` from the shared `results.py`, so its numbers are live.
Content: the R_S screening-tool reframe on the Vogel (1985) benchmark building
(scenario `vogel_six_storey`), the R_S-vs-code-ALP column agreement
(Spearman rho = 0.91, `analysis/importance.compare_column_alp`), and the
base-fixity design-variant demo (`analysis/design_variants`). Figures
`fig_alp_agreement.png` and `fig_design_variants.png`. NOT submitted (blocked
until LAJSS resolves); no cover letter drafted.

## Per-journal specifics

- **LAJSS** — `LAJSS/build_docx_lajss.py`. Submitted; in peer review.
- **IngInv** — `IngInv/build_docx_ingeinv.py`, template `IngInv/Template_IngInv.docx`.
  Bilingual EN body plus ES titulo / resumen / palabras clave; numbered `[n]`
  IEEE references; CRediT, conflicts, and data-availability statements.
  Submitted under the Civil and Sanitary Engineering section; a signed License
  Agreement was uploaded. The signed license and its `fill_license.py` are
  gitignored (they carry private identifying data) and are not in this repo.
- **ACM** — `ACM/build_docx_acm.py`, template `ACM/acmtempl.docx`. Numbered `[n]`
  Elsevier references; minimum 8 pages and an even final page count; submit at
  acm.kme.zcu.cz; requires PDF plus Word source plus separate 300 dpi figures;
  no cover letter or license form.
- **RCUC / RDLC** — `RDLC/build_docx.py`. Original target.
- **Ingeniare** — `Ingeniare/` holds `fix_ingeniare.py`, `build_carta.py`,
  `paper_ingeniare.docx`, `Carvajal_Ingeniare_manuscript_v3.docx`, plus the
  carta and Modelo.

## Conventions

Follow the repo-root `CLAUDE.md`: English, plain ASCII (no emojis, avoid
em-dashes in generated prose), numbers reproducible from the live analysis.
When a submission status changes, update the table above and the matching
per-journal note in the same edit.
