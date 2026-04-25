#!/usr/bin/env bash
# Part 3 — Local ancestry inference on simulated admixed haplotypes.
#
# Five chimeric haplotypes (CHIM_00#1 ... CHIM_04#1) were stitched from real
# HPRCv2 AFR and EUR assemblies along ground-truth tract boundaries. The
# reference panel contains 10 AFR + 10 EUR haplotypes (from HPRCv2, none of
# them used as donors of the chimeras).
#
# Region: chr12:50,000,001-120,000,000 (70 Mb spanning the middle of the long
# q arm of chr12, well clear of the centromere and both telomeres). Within
# this region the five chimeras have 3-7 ground-truth ancestry breakpoints
# each, so boundary recovery can be judged visually.
# Window: 10 kb.
# Ground truth: data/ground_truth_tracts.tsv

set -euo pipefail

mkdir -p solutions

bin/ancestry \
    --similarity-file data/identity_chr12_admix.tsv \
    --query-samples data/queries.txt \
    --populations data/populations.tsv \
    --region chr12:50000001-120000000 \
    --window-size 10000 \
    --auto-configure \
    --identity-floor 0.9 \
    --output solutions/ancestry_segments.tsv \
    --posteriors-output solutions/ancestry_posteriors.tsv

echo
echo "Ancestry segments per chimera (first lines):"
column -t -s $'\t' solutions/ancestry_segments.tsv | awk 'NR<=20'
echo
echo "Total decoded segments: $(($(wc -l < solutions/ancestry_segments.tsv) - 1))"
echo
echo "Ground truth (for comparison):"
column -t -s $'\t' data/ground_truth_tracts.tsv
