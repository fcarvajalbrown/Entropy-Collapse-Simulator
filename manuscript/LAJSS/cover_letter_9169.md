Submission 9169: "An Entropy Robustness Index for Planar Steel Frames: Formulation, Verified Implementation, and Comparison with Standard Collapse Criteria"

Response to the reviewer

Dear Editor,

Thank you for sending the reviewer's report. I have revised the manuscript to answer each of the nine points, and every change in the revised file is set in red. Below, each comment is quoted and followed by what I changed and where. I also corrected three errors the reviewer did not raise; they are listed at the end.

Comment 1. "What is an "Euler-Bernoulli solver"?"

That phrase is not standard terminology, and it is gone from the manuscript. The analysis is now described as a first-order linear-elastic finite-element model of planar frames with Euler-Bernoulli beam elements (Abstract and Section 2.4), and the independent check as a finite-element program with three degrees of freedom per node (Section 2.4). The same wording replaces "solver" in the rest of Section 2.4.

Comment 2a. "Is this relevant to the robustness index you are proposing? "verified to floating-point precision""

It is relevant, but the manuscript never said why, and the phrase told the reader nothing useful. R_S is computed directly from the member strain energies, so any error in those energies is an error in the index. Section 2.4 now opens by saying so. The phrase has been removed from the Abstract and the Conclusion, and Table 1 now reports the measured errors instead of "0.0000%". Every displacement and strain-energy error is below 4 × 10^-13 %.

Comment 2b. "What is this? "And I make no claim of early warning.""

The sentence was unclear, and I rewrote it (Introduction, last paragraph). It now says what an early-warning claim would be, namely that a falling entropy forecasts collapse, and states that the paper does not make it. The reason comes from the paper's own result: in the incremental analysis of Section 3.4 the entropy criterion fires at the same load as the energy criterion and one load step after the DCR criterion (Table 4), never before them. So the entropy drop is presented as a marker of load-path localization at the moment it happens. I also removed the sentence on feedback-amplified systems, which had no reference.

Comment 3. "Equations are unstructured, which makes them difficult to read;"

I agree. The old Equation (1) put three formulas on one line in plain text. They are now three numbered equations, (1) to (3), and the index definition is split into Equations (4) and (5). All of them are typeset with explicit summation limits, and each group is followed by a "where" clause that defines every symbol. Inline mathematics is also written with proper symbols: the entropy drop ΔH_k, the compliance U/λ², the worst case min_k H_k, the stiffness system Ku = F and the compliance importance I_k, which Section 3.3 now defines in full. The figure axes that used to read "dH_k" now show ΔH_k as well.

Comment 4. "What is an active member? In the load or the strain energy on a member is zero, is this member active or not?"

Section 2.1 now defines it. A member is active if it is still part of the structure, that is, it has not been removed in the alternate-load-path procedure and has not been dropped by the failure criterion during the incremental analysis. A member that carries no load is still active. Its strain energy and its share p_i are zero, it adds nothing to S under the convention 0 ln 0 = 0, and it still counts in N, so it lowers the normalized entropy H. Every result in the paper was computed this way, so no result changes.

Comment 5. "What is this? "The four criteria are read off one incremental analysis.""

The sentence was idiomatic, and the paragraph around it did not describe the procedure. I rewrote Section 2.3 to state it in full. The load factor starts at 1.0 and rises by a fixed step (0.3 for the truss of Section 3.4). At each step the current structure is analysed and four quantities are recorded; members whose combined stress reaches the yield stress are removed before the next step; and each criterion fires at the first step where its quantity crosses its limit. The limits are now given: a translation of 0.05 m, a DCR of 1.0, a compliance of three times the intact value, and an entropy drop larger than a fixed numerical floor of 10^-3 nats. The text also says plainly that the entropy test uses this fixed floor, which does not depend on the structure, and that a z-score rule with a cutoff of three standard deviations applies only when the earlier drops vary by more than the floor. Before the first failure in a linear-elastic analysis, they never do.

Comment 6. "What are all the zeros in Table 1?"

They were verification errors printed to four decimal places, so values around 10^-13 % showed up as 0.0000%, and the table did not say what they measured. Table 1 now gives the measured values (see Comment 2a), and its caption defines each entry: the relative difference between the computed and the reference value, in percent, where the reference is the closed-form solution for the analytical cases and an independent finite-element program for the independent cases. An entry of 0 now means the two values are identical. I also restored the full name of the truss case, which had been cut off.

Comment 7. "Where are the illustrations of the frames and beams addressed in the study, to give your reader to opportunity to interpret your results? Member IDs in Figure 1 make no sense unless a figure shows their location!"

The reviewer is right, and this was a real omission. A new Figure 1 at the start of Section 3.1 draws the three benchmark structures (the single-span beam, the 2-bay 3-story moment frame and the 6-panel X-braced truss bridge) with their dimensions, supports, loads and the number of every member. Those member numbers are the ones on the horizontal axis of the member-criticality figure (the old Figure 1, now Figure 2), so each bar can be found on its structure. The drawing comes straight from the model definitions used in the analysis. The other figures are renumbered to match. While drawing it I also corrected the name of the first benchmark: it is a single span pinned at both ends and modelled with two elements, not a two-span beam, and it is now described as a beam with no alternate load path rather than as non-redundant.

Comment 8. "Especially for the analyzed truss, I have no idea how you are achieving redundancy going from 1 to 6!"

Section 3.2 now explains it step by step. The sweep starts from a plain 6-panel Pratt truss with 25 bars, 14 joints and 3 support reactions. Since m + r = 25 + 3 = 28 = 2j, it is statically determinate. A counter-diagonal, running from bottom joint i to top joint i + 1, is then added to one panel at a time, from left to right. Each one adds a bar and no joint, so the degree of static indeterminacy, DSI = m + r − 2j, equals the number of counter-diagonals added, from 0 to 6. The counter-diagonals are members 25 to 30 in the new Figure 1(c), and Table 3 has a new column listing which of them are present at each DSI.

Comment 9. "Conclusions should be self-contained: which benchmarks? "R_S separates the benchmarks (0.00, 0.72, 0.79)""

I agree, and the Conclusion has been rewritten to stand on its own. It now names the structure behind each number: R_S is 0.00 for the single-span beam, which has no alternate load path, 0.72 for the 2-bay 3-story steel moment frame and 0.79 for the 6-panel X-braced truss bridge, and on the truss it rises from 0.00 to 0.79 as the degree of static indeterminacy goes from 0 to 6. The Spearman correlations are now attributed to the moment frame (-0.34) and the truss (+0.54). The criteria comparison is also stated more precisely than before. The entropy criterion fires at the same load factor as the energy criterion (8.80) and one load step after the DCR criterion (8.50); the original text said it fired at the same load as both.

Other corrections

Three errors that the reviewer did not raise are also fixed. The author affiliation now reads "M.Sc. in Numerical Simulation in Engineering (Universidad Politécnica de Madrid, Spain), Independent Researcher, Santiago (Chile)". Section 3.1 said the truss had the sharper worst-case localization, but Table 2 shows it is the moment frame (H_k = 0.390 against 0.737), and the sentence now says so. The Introduction said that R_S, like the redundancy matrix, is load-independent. That is wrong: R_S depends on the load pattern and is invariant only to load scaling, and the sentence has been corrected.

Finally, the title in the submission record reads "A Entropy Robustness Index ...". The manuscript title is "An Entropy Robustness Index ...", and I have corrected the record to match.

Sincerely,

Felipe Carvajal Brown
M.Sc. in Numerical Simulation in Engineering (Universidad Politécnica de Madrid, Spain), Independent Researcher, Santiago (Chile)
fcarvajalbrown@gmail.com
ORCID: 0000-0002-8300-7587
