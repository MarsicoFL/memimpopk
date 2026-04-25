# `impop_k` — workshop session

Student-facing materials for a 60-minute teaching session at the
[MEMPANG pangenomics workshop](https://pangenome.github.io/MemPanG25/).

You will go from a per-window pairwise identity matrix (the output of
`impg similarity` from the previous day) to four downstream inferences
on the long arm of human chromosome 12:

1. **IBS** — explore the identity distribution of the CEPH 1463 platinum pedigree.
2. **IBD** — recover sibling- and grandparent-shared segments with a 2-state HMM.
3. **Local ancestry** — paint five admixed AFR/EUR chimeras against a 20-haplotype reference panel, then look at the per-window forward-backward posteriors and contrast them with Viterbi.
4. **Pedigree painting** — turn the same CEPH matrix into a K=4 ancestry problem with the four grandparents as "populations", and recover paternal vs. maternal homologs without ever consulting a phased VCF.

Compute time across the whole tutorial is under 10 seconds. Reading
outputs and answering interpretation questions take ~50 minutes.

## Quickstart

After cloning, from inside this directory:

```bash
# 1. Verify your system can run the materials  (~2 seconds)
bash test_setup.sh

# 2. Open the tutorial
xdg-open tutorial/tutorial.pdf      # or `open` on macOS

# 3. Follow the tutorial from Section 0
```

If `test_setup.sh` exits 0 with `OK`, you have everything you need.

## Repository layout

```
.
├── README.md                              this file
├── LICENSE                                MIT
├── test_setup.sh                          5-second smoke check
├── tutorial/
│   ├── tutorial.tex                       handout source (LaTeX)
│   └── tutorial.pdf                       compiled handout — read this
├── bin/                                   pre-built impop binaries (Linux x86_64)
│   ├── ancestry, ibd, ibs, jacquard
│   └── README.md
├── code/
│   ├── 01_explore_ibs.py                  Part 1
│   ├── 02_run_ibd.sh                      Part 2
│   ├── 03_run_ancestry.sh                 Part 3 (Viterbi + posteriors)
│   ├── 04_plot_painting.py                Part 3 painting
│   ├── 05_paint_pedigree.sh               Part 4
│   ├── 06_plot_pedigree_painting.py       Part 4 plot
│   └── 07_plot_posteriors.py              Part 3 deeper dive (FB vs Viterbi)
└── data/                                  ~129 MB total
    ├── identity_chr12_pedigree.tsv        CEPH 1463, chr12:40–130 Mb, 18 haps
    ├── identity_chr12_admix.tsv           chr12:50–120 Mb, 5 chimeras + 20 panel haps
    ├── populations.tsv                    AFR/EUR panel for Part 3
    ├── queries.txt                        chimera queries for Part 3
    ├── ground_truth_tracts.tsv            simulation truth for Part 3
    ├── pedigree_populations.tsv           4 grandparents as K=4 panel for Part 4
    └── pedigree_queries.txt               10 grandchild haplotypes for Part 4
```

`solutions/` and `figures/student_*.png` are *generated* by the tutorial
commands and are listed in `.gitignore`. They will appear when you run
the scripts.

## System requirements

| | Tested |
|---|---|
| OS | Linux x86_64 (Ubuntu 20.04+, glibc ≥ 2.31) |
| Python | 3.8+ with `numpy` and `matplotlib` |
| Bash | 4+ (any modern Linux/macOS shell) |
| Disk | ~150 MB free |

The pre-built binaries in `bin/` are stripped Linux x86_64 ELFs.
**They will not run on macOS or Windows directly.** See
"Other operating systems" below.

## Other operating systems

If `test_setup.sh` reports that the binaries don't run, you have two
options.

### macOS / Linux ARM / very old glibc — build from source

```bash
git clone https://github.com/MarsicoFL/impop
cd impop
cargo build --release
cp target/release/{ancestry,ibd,ibs,jacquard} /path/to/this/clone/bin/
```

This needs Rust 1.70+ (`rustup install stable`).

### Windows — use WSL

The tutorial is bash + Python. Native Windows is not supported.
[Install WSL2 with Ubuntu](https://learn.microsoft.com/en-us/windows/wsl/install)
and run the tutorial from inside the WSL shell.

### Workshop VM

If you are at the in-person MEMPANG workshop, the provided VM is a
recent Ubuntu and runs everything out of the box.

## How to follow the tutorial

1. Run `bash test_setup.sh` — it should print `OK` and exit 0.
2. Open `tutorial/tutorial.pdf`.
3. Follow it from Section 0 (Setup) through Section 4 (Grandparent
   painting). Each part has 3–4 numbered interpretation questions.

## How the precomputed data was built

The TSVs in `data/` are subsets of full-chromosome matrices computed
by `impg similarity` on the HPRCv2 pangenome PAF
(`hprc465vschm13.aln.paf.gz`) at 10 kb resolution, taken from the
impop paper validation experiments. Reproducing them from the raw PAF
takes hours on a workstation; the subsetted versions shipped here are
ready to use.

The chimeras of Part 3 were constructed by mosaic-copying along
ground-truth tracts from real HPRCv2 assemblies. Each chimera mixes
one AFR donor (HG01884) with one of five EUR donors (HG00097, HG00099,
HG00126, HG00128, HG00133). None of the donor samples are in the
reference panel.

## Source code

The Rust source code for the four binaries in `bin/` lives at
[`MarsicoFL/impop`](https://github.com/MarsicoFL/impop). This repository
ships only the pre-built artifacts needed to run the workshop.

## Citation

If you use this material, please credit it as

> Marsico FL, *impop_k workshop*, MEMPANG pangenomics course (2026).

and link to the impop repository:
[`MarsicoFL/impop`](https://github.com/MarsicoFL/impop).

## License

MIT — see `LICENSE`.
