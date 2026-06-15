# Solver validation report

## Tier 1 - Analytical (exact closed-form)

| Case | delta_exact (m) | delta_num (m) | err % | U_exact (J) | U_num (J) | err % | Reference |
|---|---|---|---|---|---|---|---|
| Simply-supported beam | 5.208333e-02 | 5.208333e-02 | 0.0000 | 1.302083e+03 | 1.302083e+03 | 0.0000 | Gere & Goodno, Mechanics of Materials |
| Fixed-fixed beam | 1.302083e-02 | 1.302083e-02 | 0.0000 | 3.255208e+02 | 3.255208e+02 | 0.0000 | Roark's Formulas for Stress and Strain |
| Cantilever beam | 4.166667e-02 | 4.166667e-02 | 0.0000 | 4.166667e+02 | 4.166667e+02 | 0.0000 | Gere & Goodno, Mechanics of Materials |

## Tier 2 - Independent dual solver cross-check

| Frame | peak disp (production, m) | peak disp (independent, m) | err % | U production (J) | U independent (J) | err % |
|---|---|---|---|---|---|---|
| 2D Moment Frame (2-bay, 3-story) | 5.637584e-04 | 5.637584e-04 | 0.0000 | 1.578523e+02 | 1.578523e+02 | 0.0000 |
| 2D Moment Frame (3-bay, 6-story) | 1.973154e-03 | 1.973154e-03 | 0.0000 | 1.368054e+03 | 1.368054e+03 | 0.0000 |
| Redundant truss bridge (6-panel, 30 m, X-braced) | 1.119282e-02 | 1.119282e-02 | 0.0000 | 2.096334e+03 | 2.096334e+03 | 0.0000 |

## Tier 3 - Index validation (distribution, ALP, failure, stability)

| Check | Metric | Value | Pass |
|---|---|---|---|
| per-member p_i  [2D Moment Frame (2-bay, 3-st] | max |dp_i| | 1.53e-16 | yes |
| per-member p_i  [Redundant truss bridge (6-pa] | max |dp_i| | 3.61e-16 | yes |
| post-removal ALP (rm 8) [2D Moment Frame (2] | U err % | 0.0000 | yes |
| post-removal ALP (rm 3) [Redundant truss br] | U err % | 0.0000 | yes |
| failure criterion (cantilever M*c/I) | stress err % | 0.0000 | yes |
| stability: stable removal stays stable | is_stable | True | yes |
| stability: determinate-truss removal = mechanism | mechanism | True | yes |
