# LAJSS 9169: response to the reviewer (draft)

Working draft of the point-by-point cover letter. Each reply is settled with
Felipe one comment at a time; the final letter gets the humanizer and
felipe-outward-writing pass before it is sent. Changes in the manuscript are
marked `{+like this+}` in `revision_9169.md` and will be set in colour.

## Comment 1

> What is an "Euler-Bernoulli solver"?

The phrase was not standard terminology and has been removed everywhere. The
manuscript now describes the analysis as a first-order linear-elastic
finite-element model of planar frames with Euler-Bernoulli beam elements
(Abstract, Section 2.4), and the independent check as a finite-element program
with three degrees of freedom per node (Section 2.4). The same wording replaces
"solver" elsewhere in Section 2.4.

## Comment 2a

> Is this relevant to the robustness index you are proposing? "verified to floating-point precision"

It is relevant, but the manuscript did not say why, and the phrase itself was
not useful to the reader. R_S is computed directly from the member strain
energies, so an error in those energies is an error in the index. Section 2.4
now opens by stating this. The phrase has been removed from the Abstract and the
Conclusion, and Table 1 now gives the measured errors instead of "0.0000%":
every displacement and strain-energy error is below 4 × 10^-13 %.

## Comment 2b

> What is this? "And I make no claim of early warning."

The sentence was unclear and has been rewritten (Introduction, last
paragraph). It now states what an early-warning claim would be, that a falling
entropy forecasts collapse, and that the paper does not make it. The reason is
the paper's own result: in the incremental analysis of Section 3.4 the entropy
criterion fires at the same load as the energy criterion and one load step after
the DCR criterion (Table 4), never before them. The entropy drop is therefore
presented as a marker of load-path localization when it occurs. The sentence on
feedback-amplified systems, which carried no reference, has been removed.

## Comment 3

> Equations are unstructured, which makes them difficult to read;

Agreed. The former Equation (1), which held three formulas on one line in plain
text, is now three numbered equations (1) to (3), and the index definition is
split into Equations (4) and (5). All are set as typeset equations with explicit
summation limits, and each group is followed by a "where" clause defining every
symbol. Inline mathematics (the entropy drop ΔH_k, the compliance U/λ², the
worst case min_k H_k, the stiffness system Ku = F and the compliance importance
I_k) is now written with proper symbols, and I_k is defined in full in
Section 3.3.

## Comment 4

> What is an active member? In the load or the strain energy on a member is zero, is this member active or not?

Section 2.1 now defines it. A member is active if it is still part of the
structure: it has neither been removed in the alternate-load-path procedure nor
dropped by the failure criterion during the incremental analysis. A member that
carries no load is still active. Its strain energy and its share p_i are zero,
it adds nothing to S under the convention 0 ln 0 = 0, and it still counts in N,
so it lowers the normalized entropy H. This is how the index was computed for
every result in the paper, so no result changes.

## Comment 5

> What is this? "The four criteria are read off one incremental analysis."

The sentence was idiomatic and the paragraph did not describe the procedure.
Section 2.3 has been rewritten to state it in full: the load factor starts at
1.0 and rises by a fixed step (0.3 for the truss of Section 3.4); at each step
the current structure is analysed and four quantities are recorded; members
whose combined stress reaches the yield stress are removed before the next
step; and each criterion fires at the first step where its quantity crosses its
limit. The limits are now given: a translation of 0.05 m, a DCR of 1.0, a
compliance of three times the intact value, and an entropy drop more than three
standard deviations below the mean of the earlier drops. The text also now says
plainly that the entropy test does use a cutoff, but a statistical one that does
not depend on the structure.

## Comment 6

> What are all the zeros in Table 1?

They were verification errors printed to four decimal places, so values of the
order of 10^-13 % appeared as 0.0000%, and the table did not say what they
measured. Table 1 now gives the measured values (see the reply to Comment 2a),
and its caption defines each entry: the relative difference between the
computed and reference values, in percent, where the reference is the
closed-form solution for the analytical cases and an independent finite-element
program for the independent cases. An entry of 0 now means the two values are
identical. The truncated case name of the truss has also been restored.

## Comment 7

> Where are the illustrations of the frames and beams addressed in the study, to give your reader to opportunity to interpret your results? Member IDs in Figure 1 make no sense unless a figure shows their location!

Agreed; this was a real omission. A new Figure 1 at the start of Section 3.1
draws the three benchmark structures (the single-span beam, the 2-bay 3-storey
moment frame and the 6-panel X-braced truss bridge) with their dimensions,
supports, loads and the number of every member. The member numbers are the
ones on the horizontal axis of the member-criticality figure (former Figure 1,
now Figure 2), so each bar can be located on the structure. The figure is drawn
directly from the model definitions used in the analysis. The remaining figures
have been renumbered accordingly. While preparing it I also corrected the name
of the first benchmark: it is a single span pinned at both ends and modelled
with two elements, not a two-span beam, and it is now described as a beam with
no alternate load path rather than as non-redundant.

## Comment 8

> Especially for the analyzed truss, I have no idea how you are achieving redundancy going from 1 to 6!

Section 3.2 now explains it step by step. The sweep starts from a plain 6-panel
Pratt truss with 25 bars, 14 joints and 3 support reactions; since
m + r = 25 + 3 = 28 = 2j, it is statically determinate. A counter-diagonal,
running from bottom joint i to top joint i + 1, is then added to one panel at a
time from left to right. Each adds one bar and no joint, so the degree of static
indeterminacy, DSI = m + r − 2j, equals the number of counter-diagonals added,
from 0 to 6. The counter-diagonals are members 25 to 30 in the new Figure 1(c),
and Table 3 has a new column listing which of them are present at each DSI.

## Comment 9

> Conclusions should be self-contained: which benchmarks? "R_S separates the benchmarks (0.00, 0.72, 0.79)"

Agreed. The Conclusion has been rewritten so that it can be read on its own.
It now names the structure behind each number: R_S is 0.00 for the single-span
beam, which has no alternate load path, 0.72 for the 2-bay 3-story steel moment
frame and 0.79 for the 6-panel X-braced truss bridge; on the truss it rises from
0.00 to 0.79 as the degree of static indeterminacy goes from 0 to 6. The
Spearman correlations are now attributed to the moment frame (-0.34) and the
truss (+0.54). The criteria comparison is also stated more precisely than
before: the entropy criterion fires at the same load factor as the energy
criterion (8.80) and one load step after the DCR criterion (8.50), whereas the
original text said it fired at the same load as both.
