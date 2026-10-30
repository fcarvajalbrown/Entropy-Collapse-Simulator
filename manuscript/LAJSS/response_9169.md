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
