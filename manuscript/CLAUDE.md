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

Last updated: 2026-08-29.

Exclusivity: every journal submission affirms the paper is not under review
elsewhere. It is the same paper each time, so only ONE journal may hold it at a
time. Submit sequentially, never in parallel.

| Journal | Folder | Status (2026-08-29) |
|---|---|---|
| LAJSS (Latin American Journal of Solids and Structures) | `LAJSS/` | ALIVE. Submission 9169 went to peer review and returned "revisions required" on 2026-08-29. The 2026-07-05 "no longer under consideration" entry was wrong. See the revision block below. |
| Ingenieria e Investigacion (UNAL, Colombia) | `IngInv/` | Submission 128239 acknowledged; exact status unconfirmed - verify it is closed/withdrawn, because LAJSS now demonstrably holds the same paper. |
| Applied and Computational Mechanics (Univ. West Bohemia, Czech) | `ACM/` | DESK-REJECTED - editorial priority screen, no peer review, no actionable critique. Dead; build kept for reference. |
| Revista de la Construccion / RCUC (PUC Chile) | `RDLC/` | DESK-REJECTED - discipline / novelty / general-significance screen; explicitly "not necessarily a reflection of the quality of your research." No peer review. Dead. |
| Ingeniare (Chile) | `Ingeniare/` | Staged in repo, not submitted. |

Notes on the two rejections: both were editorial desk-rejects on venue fit
(priority; discipline / novelty / significance), not peer review. Neither
returned reviewer comments, so there is nothing substantive to revise off them
before the next submission.

## LAJSS revision round (submission 9169)

Paper 1 is alive at LAJSS and is the only live peer-reviewed track. Editor:
Pablo Munoz (section editor PABLO ANDRES MUNOZ ROJAS). Dashboard:
`https://lajss.org/LAJSS/dashboard/mySubmissions?workflowSubmissionId=9169`.

Sequence: "Send to Review", then a decision requiring revisions received
2026-08-29. LAJSS allows 21 days, so the resubmission deadline as stated is
**2026-09-19**.

What LAJSS asks for at resubmission:
- A revised manuscript with every change in a different font colour.
- A separate cover letter stating all changes made, answering the reviewers
  point by point.

**Reports received (blocker closed).** The reports never showed on the author
side; Pablo Munoz pasted them into the discussion thread opened on 2026-08-29.
They are kept verbatim in `LAJSS/reviewer_comments_9169.md`: one reviewer, nine
points. On timing he wrote only "if you need more time, please inform me": no
extension has been granted in writing. As of 2026-10-30 the submission shows as
active with no deadline on the dashboard, and an "action required" notice
arrived that day. Ask Pablo for the time needed before resubmitting late.

**Open: abstract overstates the criteria result.** The Abstract says the
entropy criterion "reaches the collapse onset at the same load as the
resistance and energy criteria". Table 4 has DCR at 8.50 and entropy and
energy at 8.80, one load step apart, and Section 3.4 says entropy lags DCR.
Settle the wording with Felipe before the resubmission.

**Format.** LAJSS publishes only a Word template; there is no LaTeX one, and
the upload must be a PDF laid out like a published article. Felipe chose to
stay in Word. `LAJSS/LAJSS - Template.docx` comes from
`https://www.lajss.org/index.php/LAJSS/libraryFiles/downloadPublic/22`
(zip, also ships Calibri). Revisions go into `LAJSS/revision_9169.md` with
`{+ +}` marks first; `build_docx_lajss.py` is ported only after all nine
comments are answered.

**Build port: keep Vogel out of Table 1.** `results.py` now returns the
Vogel (1985) frame among the independent cases (added for paper 2).
`build_docx_lajss.py` fills Table 1 from every independent case, so the port
must filter Vogel out and must stop truncating case names (`[:34]`).

**Beam renamed (settled).** The paper now calls the `frame_2d_simple` case a
"single-span beam" with "no alternate load path": it is one 10 m span pinned
at both ends (axially indeterminate to degree 1), so "two-span" and
"non-redundant" were both wrong. The code's `name="Two-span beam"` still
needs the same rename in the build port.

**Title typo in the OJS record.** LAJSS holds the title as "A Entropy Robustness
Index ...". The manuscript and `LAJSS/build_docx_lajss.py` are correct with "An
Entropy Robustness Index ...", so the error is in the submission metadata only.
Fix it at Publication > Title & Abstract, and note the correction in the cover
letter so the record and the manuscript do not diverge.

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
`fig_alp_agreement.png` and `fig_design_variants.png`. Conclusions are prose and
Equations (1)-(2) are native Word (OMML) objects.

**Correction (2026-08-29):** the premise below turned out to be false. Paper 1
was still at LAJSS and has since come back from peer review with revisions
required. This does not affect paper 2's eligibility (different manuscript), but
the "LAJSS is clear" reasoning recorded on 2026-07-05 was wrong.

**Submission status (2026-07-05):** Felipe confirms paper 1 is no longer under
consideration at LAJSS, so the single-venue exclusivity block is clear. Paper 2
is being submitted to Revista de la Construccion. Comments-for-the-Editor text
was prepared; it makes no "not under review elsewhere" claim in its body (that
affirmation is the separate OJS checkbox). The "second RDLC rejection -> pivot
venue" contingency above still stands.

## Per-journal specifics

- **LAJSS** — `LAJSS/build_docx_lajss.py`. Peer-reviewed, revisions required
  2026-08-29; see the LAJSS revision round section above for the deadline, the
  coloured-text requirement, and the reviewer reports.
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
