# `impopk` — workshop materials

Materials for the hands-on session on relatedness and
ancestry inference from pangenome-derived alignments, delivered at
the [MEMPANG pangenomics
workshop](https://pangenome.github.io/MemPanG26/). The session
demonstrates how a per-window pairwise identity matrix produced by
`impg similarity` can be turned into four downstream inferences: (1)
identity-by-state structure, (2) identity-by-descent segments, (3) local
ancestry, and (4) pedigree painting, without depending on variant calling, phasing or vcf files.

## Scope

All four parts operate on the long (q) arm of human chromosome 12,
in the CEPH 1463 platinum pedigree (Parts 1, 2, 4) or in five
chimeric AFR/EUR query haplotypes painted against an HPRCv2 reference
panel (Part 3). The underlying tooling is the `impop_k` suite
(binaries `ibs`, `ibd`, `ancestry`, `jacquard`).

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
│   └── slides.pdf                15-minute Beamer talk
├── tutorial/
│   └── mempang_tutorial.pdf     participant handout (~50 min)
├── code/
│   ├── 01_explore_ibs.py         Part 1
│   ├── 02_run_ibd.sh             Part 2
│   ├── 03_run_ancestry.sh        Part 3 — Viterbi + posteriors
│   ├── 04_plot_painting.py       Part 3 painting figure
│   ├── 05_paint_pedigree.sh      Part 4
│   ├── 06_plot_pedigree_painting.py
│   ├── 07_plot_posteriors.py     Part 3 — forward–backward vs. Viterbi
│   └── _svgplot.py               stdlib SVG helpers used by 01/04/06/07
├── data/                         precomputed identity matrices and panels (~129 MB)
├── bin/                          pre-built `impop_k` binaries (Linux x86_64)
├── test_setup.sh                 environment smoke check
└── LICENSE                       MIT
```

Generated artifacts (`solutions/`, `figures/*.svg`) appear when the
tutorial scripts are executed and are excluded from version control
via `.gitignore`.

## Quick start

From the repository root:

```bash
bash test_setup.sh
```

A successful invocation prints `OK` and exits 0. The script verifies
that the pre-built binaries execute on the host, that `python3` (≥ 3.8)
is available, and that the data files and scripts are in place. The
plotting scripts use only the Python standard library, so no `pip`
install is required. Any failure prints the missing component and a
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
(SVG plots).

## System requirements

| Component | Tested configuration |
|-----------|----------------------|
| Operating system | Linux x86_64, glibc ≥ 2.31 (Ubuntu 20.04+) |
| Python | 3.8+ (stdlib) |
| Shell | bash 4+ |
| Disk | ~150 MB free |
| Memory | < 300 MB peak resident set across all steps |

The plotting scripts emit `figures/*.svg` files; open them in any
browser, image viewer (Eye of GNOME, Preview.app), or Inkscape.

The shipped binaries in `bin/` are stripped Linux x86_64 ELFs. On
macOS or ARM Linux they should be rebuilt from source (ask for the code: franco.lmarsico@gmail.com).

This requires Rust 1.70+. 

## Construction of the precomputed data

The identity matrices in `data/` are subsets of full-chromosome
matrices computed by `impg similarity` on the HPRCv2 pangenome PAF
(`hprc465vschm13.aln.paf.gz`) at 10 kb window resolution. Reproducing
them from the raw alignment file requires several hours on a
workstation.

The five chimeric query haplotypes used in Part 3 were constructed by
mosaic-copying along ground-truth tracts from real HPRCv2 assemblies.
Each chimera combines one AFR donor (HG01884) with one of five EUR
donors (HG00097, HG00099, HG00126, HG00128, HG00133). None of the
donor samples appear in the reference panel, so inference is
performed against a held-out set of haplotypes representative of,
but not identical to, the donor populations. Breakpoints were derived from slim simulations.

Released under the MIT License (see `LICENSE`).
