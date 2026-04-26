# `impop_k` — workshop materials

Teaching materials accompanying a 60-minute hands-on session on
relatedness and ancestry inference from pangenome-derived alignments,
delivered at the [MEMPANG pangenomics
workshop](https://pangenome.github.io/MemPanG26/). The session
demonstrates how a per-window pairwise identity matrix produced by
`impg similarity` can be turned into four downstream inferences —
identity-by-state structure, identity-by-descent segments, local
ancestry, and pedigree painting — without an intermediate variant
catalog.

The repository is intended both as (i) a self-contained workshop kit
that participants and instructors can deploy on a laptop and (ii) a
reproducible reference for anyone teaching or extending the same
material.

## Scope

All four parts operate on the long (q) arm of human chromosome 12,
in the CEPH 1463 platinum pedigree (Parts 1, 2, 4) or in five
chimeric AFR/EUR query haplotypes painted against an HPRCv2 reference
panel (Part 3). The underlying tooling is the `impop_k` suite
(binaries `ibs`, `ibd`, `ancestry`, `jacquard`); the slides cover the
modelling choices and the tutorial provides a guided execution path.

| Part | Method | Inference target |
|------|--------|------------------|
| 1 | IBS distribution analysis | structure of the per-window identity matrix |
| 2 | 2-state HMM (Viterbi + Baum–Welch) | IBD segments between haplotype pairs |
| 3 | K-state HMM with parametric emissions | local ancestry of admixed haplotypes |
| 4 | K=4 HMM with grandparental panels | paternal vs. maternal homolog assignment |

## Repository contents

```
.
├── presentation/
│   ├── slides.tex                15-minute Beamer talk (sources)
│   └── slides.pdf                compiled deck
├── tutorial/
│   ├── tutorial.tex              participant handout (sources)
│   ├── tutorial.pdf              compiled handout (~50 min)
│   ├── tutorial_solutions.tex    annotated answer key (sources)
│   └── tutorial_solutions.pdf    compiled answer key (instructors)
├── code/
│   ├── 01_explore_ibs.py         Part 1
│   ├── 02_run_ibd.sh             Part 2
│   ├── 03_run_ancestry.sh        Part 3 — Viterbi + posteriors
│   ├── 04_plot_painting.py       Part 3 painting figure
│   ├── 05_paint_pedigree.sh      Part 4
│   ├── 06_plot_pedigree_painting.py
│   ├── 07_plot_posteriors.py     Part 3 — forward–backward vs. Viterbi
│   └── _make_hmm_diagram.py      builds the HMM schematic used in slides
├── data/                         precomputed identity matrices and panels (~129 MB)
├── figures/                      static figures referenced by the LaTeX sources
├── bin/                          pre-built `impop_k` binaries (Linux x86_64)
├── test_setup.sh                 environment smoke check
└── LICENSE                       MIT
```

Generated artifacts (`solutions/`, `figures/student_*.png`) are
produced when the tutorial scripts are executed and are excluded from
version control via `.gitignore`.

## Recommended reading order

For a first encounter with the material, the suggested sequence is:

1. `presentation/slides.pdf` — conceptual framing of the four
   inference tasks and the underlying HMM architecture.
2. `tutorial/tutorial.pdf` — guided execution of the seven scripts in
   `code/`, with interpretation prompts.
3. `tutorial/tutorial_solutions.pdf` — discussion of the expected
   outputs and the calibration / model-mismatch points raised by the
   prompts (intended primarily for instructors).

Total compute time across the entire tutorial is on the order of
ten seconds on a recent laptop; the remaining session time is spent
reading outputs and addressing the interpretation questions.

## Quick start

From the repository root:

```bash
bash test_setup.sh
```

A successful invocation prints `OK` and exits 0. The script verifies
that the pre-built binaries execute on the host, that `python3` with
`numpy` and `matplotlib` is available, and that the data files and
scripts are in place. Any failure prints the missing component and a
remediation hint.

To execute the tutorial pipeline end-to-end:

```bash
python3 code/01_explore_ibs.py
bash    code/02_run_ibd.sh
bash    code/03_run_ancestry.sh
python3 code/04_plot_painting.py
python3 code/07_plot_posteriors.py
bash    code/05_paint_pedigree.sh
python3 code/06_plot_pedigree_painting.py
```

Outputs accumulate under `solutions/` (TSV tables) and `figures/`
(PNG plots).

## System requirements

| Component | Tested configuration |
|-----------|----------------------|
| Operating system | Linux x86_64, glibc ≥ 2.31 (Ubuntu 20.04+) |
| Python | 3.8+, with `numpy` and `matplotlib` |
| Shell | bash 4+ |
| LaTeX (optional) | `pdflatex` with `beamer`, `metropolis`, `tcolorbox`, `tikz` (only required to recompile the PDFs) |
| Disk | ~150 MB free |
| Memory | < 300 MB peak resident set across all steps |

The shipped binaries in `bin/` are stripped Linux x86_64 ELFs. On
macOS or ARM Linux they should be rebuilt from source:

```bash
git clone https://github.com/MarsicoFL/impop
cd impop && cargo build --release
cp target/release/{ancestry,ibd,ibs,jacquard} /path/to/this/repo/bin/
```

This requires Rust 1.70+. Windows users are advised to operate from
within [WSL2 with
Ubuntu](https://learn.microsoft.com/en-us/windows/wsl/install). The
in-person MEMPANG workshop provides a virtual machine on which all
materials run without additional setup.

## Construction of the precomputed data

The identity matrices in `data/` are subsets of full-chromosome
matrices computed by `impg similarity` on the HPRCv2 pangenome PAF
(`hprc465vschm13.aln.paf.gz`) at 10 kb window resolution. Reproducing
them from the raw alignment file requires several hours on a
workstation; the subsetted matrices distributed here are derived from
the validation experiments of the `impop_k` paper and are sufficient
for all four tutorial parts.

The five chimeric query haplotypes used in Part 3 were constructed by
mosaic-copying along ground-truth tracts from real HPRCv2 assemblies.
Each chimera combines one AFR donor (HG01884) with one of five EUR
donors (HG00097, HG00099, HG00126, HG00128, HG00133). None of the
donor samples appear in the reference panel, so inference is
performed against a held-out set of haplotypes representative of —
but not identical to — the donor populations.

## Reproducing the pre-built PDFs

```bash
cd presentation && pdflatex slides.tex
cd ../tutorial  && pdflatex tutorial.tex && pdflatex tutorial_solutions.tex
```

Two passes per document are sufficient for cross-references.

## Citation

If these materials are reused for teaching or derived work, please
cite as

> Marsico, F. L. *impop_k workshop*, MEMPANG pangenomics course (2026).

and the underlying tool repository:
[`MarsicoFL/impop`](https://github.com/MarsicoFL/impop).

## License

Released under the MIT License (see `LICENSE`).
