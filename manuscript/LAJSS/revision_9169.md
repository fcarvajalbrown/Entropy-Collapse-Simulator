# An Entropy Robustness Index for Planar Steel Frames: Formulation, Verified Implementation, and Comparison with Standard Collapse Criteria

Felipe Carvajal Brown a,* https://orcid.org/0000-0002-8300-7587

{+M.Sc. in Numerical Simulation in Engineering (Universidad Politécnica de Madrid, Spain), Independent Researcher, Santiago (Chile).+} Email: fcarvajalbrown@gmail.com

* Corresponding author

## Abstract

Structural robustness rarely reduces to a single number. Engineers judge it with displacement, resistance, or energy criteria, and each needs a limit calibrated to the structure at hand. This paper proposes the Entropy Robustness Index R_S, the Shannon entropy of the elastic strain-energy distribution averaged over the notional member removals of the alternate-load-path method. R_S sits near 1 when a frame keeps sharing energy evenly after any single loss, and falls toward 0 when a loss localizes the energy or forms a mechanism. Implemented {+in+} a first-order linear-elastic {+finite-element model of planar frames with Euler-Bernoulli beam elements,+} R_S reads 0.00, 0.72, and 0.79 for a {+single-span beam with no alternate load path+}, a redundant moment frame, and a redundant truss. It rises monotonically with the degree of static indeterminacy, and its member ranking is not a re-encoding of compliance importance. {+In an incremental analysis of the truss, the entropy criterion fires together with the energy criterion and one load step after the resistance (DCR) criterion, without a threshold calibrated to the structure.+}

## Keywords

structural robustness; progressive collapse; alternate load path; Shannon entropy; redundancy

## 1 INTRODUCTION

Progressive collapse is the spread of local damage into a failure out of proportion to what caused it. Guidance against it, from the GSA (2003) guidelines to Unified Facilities Criteria UFC 4-023-03 (Department of Defense, 2016), is built around the alternate load path (ALP) method: remove a primary load-carrying member, re-analyse the structure, and check whether the members that remain can carry the load along a new path. What that check should measure is less settled. Reviewing collapse criteria for reinforced-concrete frames under column removal, Feng et al. (2024) compare displacement, resistance, and energy criteria and find no consensus among them. Each one rests on a calibrated limit, whether a drift or chord-rotation limit, an acceptance demand-capacity ratio, or an energy capacity.

The idea that an even internal distribution signals robustness is not new, and it has been expressed in information-theoretic terms before. The redundancy matrix describes how static indeterminacy is spread across members and argues that robust structures distribute redundancy homogeneously (von Scheven et al., 2021); like R_S it is threshold-free, {+but unlike R_S, which depends on the load pattern and is invariant only to load scaling, it is load-independent.+} Nafday (2011) casts structural integrity in information-theoretic terms, and event-oriented system analysis defines redundancy directly as the Shannon entropy of operational modes (Ziha, 2000). In network science the Shannon entropy of a load-flow distribution is an established measure of fragility to cascading failure (Koc et al., 2013). System-reserve redundancy and robustness indices for structures are well developed (Frangopol and Curley, 1987; Ghosn et al., 2010), from which R_S differs by measuring the entropy of the elastic strain-energy distribution rather than a reserve- or reliability-capacity ratio. Strain-energy member-importance coefficients are also standard (Lin et al., 2019). Against this background the present index is a specific recombination rather than a new principle: it is the entropy of the member-level elastic strain-energy distribution, evaluated under code-style single-member removal, on planar frames. What is genuinely new is the evidence in the sections below that this particular construction (i) tracks the degree of static indeterminacy, (ii) is not reducible to compliance-based importance, and (iii) reaches the collapse onset with no calibration, placed side by side with the standard criteria on one analysis.

Two caveats belong here at the start. Strain-energy entropy is itself old, so what is new is the robustness formulation and the evidence, not the use of entropy. {+A falling entropy could be read as a warning that collapse is coming, and this paper does not make that claim. In the incremental analysis of Section 3.4 the entropy criterion fires at the same load as the energy criterion and one load step after the DCR criterion, never before them, so the entropy drop is read here as a marker of load-path localization at the moment it happens, not as a forecast of it.+}

## 2 MATERIALS AND METHODS

### 2.1 strain-energy entropy

At a given load, let U_i be the elastic strain energy in active member i, with N active members in all. {+A member is active if it is still part of the structure: it has neither been removed in the alternate-load-path procedure nor dropped by the failure criterion during the incremental analysis. A member that carries no load is still active. Its strain energy U_i and share p_i are zero, it adds nothing to S under the convention 0 ln 0 = 0, and it still counts in N, so it lowers H.+} {+Equations (1) to (3)+} give the normalized energy distribution and its Shannon entropy.

{+$$p_i = \frac{U_i}{\sum_{j=1}^{N} U_j} \qquad (1)$$+}

{+$$S = -\sum_{i=1}^{N} p_i \ln p_i \qquad (2)$$+}

{+$$H = \frac{S}{\ln N}, \qquad 0 \le H \le 1 \qquad (3)$$+}

{+where U_i is the elastic strain energy of active member i, N the number of active members, p_i the share of the total strain energy carried by member i, S the Shannon entropy and H the normalized entropy.+}

H equals 1 when the energy is shared evenly and 0 when a single member holds it all. Because the fractions p_i are ratios, S and H do not move as the load is scaled on a fixed topology. They shift only when the load path itself changes.

### 2.2 alternate load path and the robustness index

For a linear-elastic frame the redistribution after a member is lost follows exactly from dropping that member from the stiffness assembly and solving {+$\mathbf{K}\mathbf{u} = \mathbf{F}$+} again; no separate redistribution rule is needed. If the reduced stiffness matrix loses positive-definiteness, the removal has produced a mechanism. Writing H_0 for the intact normalized entropy and H_k for the survivors' entropy after removing member k (with H_k = 0 for a mechanism), the entropy drop is {+given by Equation (4)+} and the index is the mean over the removal set R, {+Equation (5)+}.

{+$$\Delta H_k = H_0 - H_k \qquad (4)$$+}

{+$$R_S = \frac{1}{|R|} \sum_{k \in R} H_k, \qquad 0 \le R_S \le 1 \qquad (5)$$+}

{+where H_0 is the normalized entropy of the intact frame, H_k that of the frame after member k is removed (H_k = 0 if a mechanism forms), and |R| the number of removals in R.+}

An R_S near 1 means the frame stays redundant under any single loss; near 0 means it does not. The index is bounded and needs no threshold tuned to the structure. Two summaries accompany it: the worst case {+$\min_k H_k$+}, and R_S restricted to the removals that stay stable, which separates how even the surviving distribution is from how many losses are survivable. A determinacy bound makes the link to redundancy precise: a statically determinate truss has R_S = 0, because removing any bar of it forms a mechanism, so R_S > 0 requires at least one degree of static indeterminacy (proof in the accompanying theory note).

### 2.3 four criteria for comparison

{+All four criteria are evaluated on the same incremental analysis, so they are compared on one sequence of structural states. The load factor $\lambda$ starts at 1.0 and rises by a fixed step (0.3 for the truss of Section 3.4). At each step the current structure is analysed and four quantities are recorded: the largest nodal translation, the largest demand-capacity ratio (DCR, the combined axial and bending stress over the yield stress $\sigma_y$) among active members, the compliance $U/\lambda^2$, and the change in entropy from the previous step. Every member whose combined stress reaches $\sigma_y$ is then removed, and the next step is analysed without it. Each criterion fires at the first step where its quantity crosses a limit: displacement when the largest translation exceeds 0.05 m, DCR when the largest DCR reaches 1.0, energy when the compliance exceeds three times its intact value, and entropy when the {+drop in the un-normalized entropy S at a step exceeds a fixed numerical floor of 10^-3 nats. The entropy test is applied from the sixth step onward. Because S is flat until the first member failure in a linear-elastic analysis, this floor is what fires the criterion in Section 3.4; a z-score rule with a cutoff of three standard deviations is used only if the earlier drops vary by more than the floor.+} Compliance is used instead of U because U grows with $\lambda^2$ on a fixed topology and would otherwise trip on load alone. The first three limits depend on the structure and must be calibrated to it; the entropy floor is a fixed numerical tolerance that does not.+}

### 2.4 implementation and verification

{+Because R_S is computed directly from the member strain energies, those energies were checked against closed-form solutions and an independent program before any index was computed.+} The tool is written in Python, with modules that talk to each other only through shared data classes. The frame element is a planar Euler-Bernoulli {+beam+} element with axial and in-plane bending stiffness; truss members are axial-only bars. One detail matters more than it looks: the local bending plane is pinned to the global X-Y plane for every member. Taking it from a per-member reference vector rotates a column's bending plane out of plane and leaves the in-plane system singular, a failure that is easy to miss because a least-squares solve still returns numbers.

Verification runs on three tiers (Table 1). The first two confirm the {+finite-element model+}: {+closed-form beam solutions are reproduced+}, and an independent {+finite-element program with three degrees of freedom per node, written from scratch,+} reproduces the redundant frames, including a larger 3-bay 6-story frame (28 nodes, 42 members). {+Every displacement and strain-energy error in Table 1 is below 4 × 10^-13 %.+} The third tier checks the quantities R_S actually depends on: the per-member distribution, the post-removal ALP states, the failure criterion, and the stability test all agree with independent calculations. The Ziemian and Ziemian (2021) steel benchmark frames are a useful external reference, but their published results are second-order (P-Delta) and sit outside what a first-order {+analysis+} can reproduce, so I cite them for future checks rather than match them here.

Table 1 {+Verification of the finite-element model. Each error is the relative difference |computed − reference| / |reference| × 100 %. The reference is the closed-form solution for the analytical cases and an independent finite-element program for the independent cases. An entry of 0 means the computed and reference values are identical. The index checks compare the quantities R_S depends on with independent calculations; all seven pass.+}

| Tier | Case | Displacement err | Strain-energy err |
|---|---|---|---|
| Analytical | Simply-supported beam | {+0+} | {+1.7 × 10^-14 %+} |
| Analytical | Fixed-fixed beam | {+0+} | {+1.7 × 10^-14 %+} |
| Analytical | Cantilever beam | {+0+} | {+1.4 × 10^-14 %+} |
| Independent | 2D Moment Frame (2-bay, 3-story) | {+1.9 × 10^-14 %+} | {+1.8 × 10^-14 %+} |
| Independent | 2D Moment Frame (3-bay, 6-story) | {+1.5 × 10^-13 %+} | {+1.2 × 10^-13 %+} |
| Independent | {+Redundant truss bridge (6-panel, 30 m, X-braced)+} | {+3.1 × 10^-13 %+} | {+3.7 × 10^-13 %+} |
| Index checks | distribution / ALP / failure / stability (7/7 pass) | - | - |

## 3 RESULTS

### 3.1 robustness index

Three frames are analysed: a {+single-span beam pinned at both ends and modelled with two elements, which has no alternate load path+}, a 2-bay 3-story steel moment frame, and a 6-panel X-braced truss bridge. Table 2 collects the index for each. {+Figure 1 shows their geometry, supports, loads and member numbering; the member numbers are those used in Figure 2.+}

[figure: fig_benchmark_geometry.png]

{+Figure 1 Geometry, supports, loads and member numbering of the three benchmark structures: (a) single-span beam, (b) moment frame, (c) truss bridge. Member numbers are boxed; dimensions in m, loads in kN.+}

Table 2 Entropy Robustness Index of the benchmark frames (computed by entropy/robustness.py).

| Frame | Intact H0 | R_S | Worst-case H_k | Mechanism fraction |
|---|---|---|---|---|
| {+Single-span beam+} | 1.000 | 0.000 | 0.000 | 100% |
| Moment frame (2-bay, 3-story) | 0.712 | 0.719 | 0.390 | 0% |
| Truss bridge (X-braced) | 0.791 | 0.786 | 0.737 | 0% |

The values fall where intuition says they should. The {+single-span+} beam has no second path: lose either member and what remains is a mechanism, so R_S = 0. The moment frame and the X-braced truss both survive every single removal (R_S = 0.72 and 0.79), but the worst single loss in the {+moment frame pulls its surviving distribution down to H_k = 0.390, a sharper localization than the worst case of the truss (H_k = 0.737).+} The larger 3-bay 6-story frame behaves the same way (R_S = 0.78), so the index is not an artefact of small models.

[figure]

{+Figure 2+} Entropy-based member criticality (entropy drop {+$\Delta H_k$+}) for the moment frame and the truss bridge.

### 3.2 R_S tracks redundancy

To test whether R_S measures redundancy rather than merely re-packaging stiffness, I sweep the truss bridge from its statically determinate single-diagonal form to its fully X-braced form. {+The sweep starts from a plain 6-panel Pratt truss with m = 25 bars (6 bottom-chord, 6 top-chord, 7 vertical and 6 diagonal), j = 14 joints and r = 3 support reactions. Since m + r = 25 + 3 = 28 = 2j, it is statically determinate. A counter-diagonal, running from bottom joint i to top joint i + 1, is then added to one panel at a time, from left to right: these are members 25 to 30 in Figure 1(c). Each adds one bar and no joint, so the degree of static indeterminacy DSI = m + r − 2j equals the number of counter-diagonals added, from 0 to 6.+} R_S rises monotonically from 0.00 at zero redundancy to 0.79 at six, while the fraction of single-loss mechanisms falls from 100% to 0% ({+Figure 3+}, Table 3). The determinate case gives exactly R_S = 0, as the determinacy bound requires.

Table 3 R_S versus degree of static indeterminacy (DSI) for the truss bridge (computed by analysis/parametric.py).

| DSI | {+Counter-diagonals+} | Members | R_S | Mechanism fraction |
|---|---|---|---|---|
| 0 | {+none+} | 25 | 0.000 | 100% |
| 1 | {+25+} | 26 | 0.194 | 77% |
| 2 | {+25-26+} | 27 | 0.340 | 59% |
| 3 | {+25-27+} | 28 | 0.470 | 43% |
| 4 | {+25-28+} | 29 | 0.583 | 28% |
| 5 | {+25-29+} | 30 | 0.680 | 13% |
| 6 | {+25-30+} | 31 | 0.786 | 0% |

[figure]

{+Figure 3+} R_S rises and the single-loss mechanism fraction falls as the truss gains redundancy.

### 3.3 the ranking is not compliance importance

A natural objection is that {+$\Delta H_k$+} just restates the compliance increase caused by removing a member. The two are related but not the same. Comparing the entropy drop with the removal-based compliance importance {+$I_k = (U^{(k)} - U_0)/U_0$, where U_0 is the total strain energy of the intact frame and $U^{(k)}$ the total after member k is removed,+} gives a Spearman rank correlation of -0.34 for the moment frame and +0.54 for the truss ({+Figure 4+}). The correlation is frame-dependent and well below one; for the moment frame the two rankings are uncorrelated to weakly negative, so the member whose loss most softens the structure need not be the one whose loss most localizes the energy. I do not claim statistical significance at this sample size; the point is that {+$\Delta H_k$+} is not a monotone re-encoding of compliance importance.

[figure]

{+Figure 4+} Entropy criticality {+$\Delta H_k$+} against compliance importance I_k; the two rankings differ (Spearman in titles).

To check that this is a property of the measure and not of one example, I repeat the comparison over an ensemble of 11 frames of varied size and type (moment frames from one to four bays and two to six stories, and redundant trusses). The rank correlation ranges from -0.56 to +0.87 with mean +0.17, and is negative for 6 of the 11 frames ({+Figure 5+}). The sign is systematic rather than random: every multi-bay moment frame gives a negative correlation, while small frames and trusses give positive ones. The entropy criticality is therefore consistently far from a monotone re-encoding of compliance importance, and for a whole class of structures it ranks members in essentially the opposite order.

[figure]

{+Figure 5+} Spearman correlation between entropy criticality and compliance importance across the frame ensemble; it stays well below 1 and is negative for the multi-bay moment frames.

### 3.4 progressive collapse and criteria comparison

Loading the truss bridge step by step to failure shows the entropy behaving as expected. While the truss is intact the entropy holds flat, since it ignores load magnitude; at the first member failure it drops as the energy crowds into fewer members, then recovers part of the way as the remaining bars take up the load ({+Figure 6+}). Table 4 lists the load factor at which each criterion first fires.

[figure]

{+Figure 6+} Normalized entropy and compliance versus load step for the truss bridge; both are flat until the first member failure, then respond together.

Table 4 First-trigger load factor of the four criteria (truss bridge, load-step 0.3; computed by analysis/criteria.py).

| Criterion | First trigger (load factor) | Threshold required |
|---|---|---|
| Displacement | 4.60 | yes (drift limit) |
| DCR | 8.50 | yes (acceptance DCR) |
| Energy (compliance) | 8.80 | yes (energy multiple) |
| Entropy | 8.80 | {+no (fixed 10^-3 nats floor)+} |

[figure]

{+Figure 7+} First-trigger load factor for the four criteria; grey bars need a calibrated threshold, the entropy bar does not.

The displacement criterion trips first and where it trips is set by the drift limit, a known soft spot of displacement-based checks. The other three agree near load factor 8.50 to 8.80. DCR catches the stress that triggers the first failure; the energy and entropy criteria catch its aftermath at the same load. I want to be precise about this: in the linear-elastic range the entropy is flat until the first removal, so the entropy criterion is a one-step-lagging, consequence-side indicator, not an onset detector that leads DCR. A step-size study confirms it: the DCR-to-entropy gap is one load step at every step size and shrinks with it. The honest claim is that entropy reaches the same conclusion as DCR and energy without a calibrated threshold, not that it sees collapse sooner.

### 3.5 limitations

The limits should be read plainly. Entropy of strain energy is old, so the new part is the robustness formulation and the evidence that it tracks redundancy, departs from compliance importance, and is calibration-free. The analysis is planar, linear-elastic, first-order, and quasi-static, which leaves out plasticity, second-order and catenary action, out-of-plane response, dynamic amplification, and connection failure. Linear-static ALP is an accepted and usually conservative code method, though for stability-governed compression members the first-order assumption can be unconservative. And the entropy drop lands with the localization rather than ahead of it, so the claim is a calibration-free measure of robustness and localization, not lead time.

## 4 CONCLUSION

R_S is a bounded redundancy measure, built from the strain-energy entropy under the alternate-load-path procedure, that {+needs no threshold calibrated to the structure and equals 0 for a statically determinate structure. For the three benchmarks it is 0.00 for the single-span beam, which has no alternate load path, 0.72 for the 2-bay 3-story steel moment frame and 0.79 for the 6-panel X-braced truss bridge. As counter-diagonals are added to the truss, taking its degree of static indeterminacy from 0 to 6, R_S rises monotonically from 0.00 to 0.79. The entropy member-criticality ranking is not a monotone re-encoding of compliance importance: the Spearman rank correlation between the two is -0.34 for the moment frame and +0.54 for the truss. In the incremental analysis of the truss, the entropy criterion fires at the same load factor as the energy criterion (8.80) and one load step after the DCR criterion (8.50), without a limit calibrated to the structure.+}

Author's Contributions: LAJSS endorses the CRediT taxonomy of contributor roles. Conceptualization, F Carvajal Brown; Methodology, F Carvajal Brown; Software, F Carvajal Brown; Validation, F Carvajal Brown; Formal analysis, F Carvajal Brown; Investigation, F Carvajal Brown; Writing - original draft, F Carvajal Brown; Writing - review & editing, F Carvajal Brown.

Data Availability: Research data is available in a repository. The source code, scenarios, and scripts that reproduce every number and figure in this article are openly available at https://github.com/fcarvajalbrown/Entropy-Collapse-Simulator (GPL-3.0-or-later).

Conflicts of Interest: The author declares no conflict of interest.

Funding: This research received no external funding.

Editor: (Filled by: Journal Staff)

References

Department of Defense (2016). Unified Facilities Criteria UFC 4-023-03: Design of Buildings to Resist Progressive Collapse, US Department of Defense.

Feng, D.C., et al. (2024). Physically-based collapse failure criteria in progressive collapse analyses of random-parameter multi-story RC structures subjected to column removal scenarios. Engineering Structures. https://doi.org/10.1016/j.engstruct.2024.119412

Frangopol, D.M., Curley, J.P. (1987). Effects of damage and redundancy on structural reliability. Journal of Structural Engineering 113(7):1533-1549.

General Services Administration (2003). Progressive Collapse Analysis and Design Guidelines for New Federal Office Buildings and Major Modernization Projects, US General Services Administration.

Ghosn, M., Moses, F., Frangopol, D.M. (2010). Redundancy and robustness of highway bridge superstructures and substructures. Structure and Infrastructure Engineering 6(1-2):257-278.

Koc, Y., Warnier, M., Kooij, R.E., Brazier, F.M. (2013). An entropy-based metric to quantify the robustness of power grids against cascading failures. Safety Science 59:126-134.

Lin, K., et al. (2019). Importance assessment of structural members based on elastic-plastic strain energy. Advances in Materials Science and Engineering 2019:8019675.

Nafday, A.M. (2011). Consequence-based structural design approach for black swan events. Structural Safety 33(1):108-114.

Shannon, C.E. (1948). A mathematical theory of communication. Bell System Technical Journal 27:379-423.

von Scheven, M., Ramm, E., Bischoff, M. (2021). Quantification of the redundancy distribution in truss and beam structures. International Journal of Solids and Structures 213:41-49. https://doi.org/10.1016/j.ijsolstr.2020.11.002

Ziemian, C.W., Ziemian, R.D. (2021). Steel benchmark frames for structural analysis and validation studies: Finite element models and numerical simulation data. Data in Brief 39:107564. https://doi.org/10.1016/j.dib.2021.107564

Ziha, K. (2000). Redundancy and robustness of systems of events. Probabilistic Engineering Mechanics 15(4):347-357.
