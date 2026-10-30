import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "manuscript"))
sys.path.insert(0, ROOT)

from results import compute_results


def main():
    R = compute_results()
    print("tier\tcase\tdisplacement_err_pct\tstrain_energy_err_pct")
    for r in R["analytical"]:
        print(f"analytical\t{r['case']}\t{r['delta_err_pct']:.3e}\t{r['U_err_pct']:.3e}")
    for r in R["independent"]:
        print(f"independent\t{r['frame']}\t{r['disp_err_pct']:.3e}\t{r['U_err_pct']:.3e}")
    for c in R["index_checks"]:
        print(f"index_check\t{c['check']}\tpass={c['pass']}")


if __name__ == "__main__":
    main()
