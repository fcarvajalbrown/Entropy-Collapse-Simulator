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
"solver" in Section 2.4 and in the Conclusion.

## Comment 2a

> Is this relevant to the robustness index you are proposing? "verified to floating-point precision"

It is relevant, but the manuscript did not say why, and the phrase itself was
not useful to the reader. R_S is computed directly from the member strain
energies, so an error in those energies is an error in the index. Section 2.4
now opens by stating this. The phrase has been removed from the Abstract and the
Conclusion, and Table 1 now gives the measured errors instead of "0.0000%":
every displacement and strain-energy error is below 4 × 10^-13 %.
