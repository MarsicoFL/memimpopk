#!/usr/bin/env bash
# Part 4 — Paint the chromosomes of the five CEPH 1463 grandchildren against
# the four grandparents as "populations".
#
# This reuses the same identity matrix as Part 2 (the CEPH pedigree,
# chr12:40-130 Mb). The trick is that the four grandparents are loaded
# as K=4 "populations" in the ancestry CLI, one per grandparent:
#
#     GP_PAT = NA12889    GM_PAT = NA12890
#     GP_MAT = NA12891    GM_MAT = NA12892
#
# The five grandchildren are the queries. The expectation is structural:
# each grandchild has two haplotypes; one of them is paternal (mosaic of
# GP_PAT + GM_PAT blocks) and the other maternal (mosaic of GP_MAT + GM_MAT).
# Cross-couple "mixing" within a single haplotype should not occur.
#
# Region: chr12:40,000,001-130,000,000 (same as Part 2).

set -euo pipefail

mkdir -p solutions

bin/ancestry \
    --similarity-file data/identity_chr12_pedigree.tsv \
    --query-samples data/pedigree_queries.txt \
    --populations data/pedigree_populations.tsv \
    --region chr12:40000001-130000000 \
    --window-size 10000 \
    --auto-configure \
    --identity-floor 0.9 \
    --output solutions/pedigree_painting.tsv

echo
echo "Painted segments per grandchild haplotype (first 20):"
column -t -s $'\t' solutions/pedigree_painting.tsv | awk 'NR<=20'
echo
echo "Total decoded segments: $(($(wc -l < solutions/pedigree_painting.tsv) - 1))"
