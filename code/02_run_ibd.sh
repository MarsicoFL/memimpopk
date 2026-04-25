#!/usr/bin/env bash
# Part 2 — IBD detection on the CEPH 1463 platinum pedigree.
#
# Five children (NA12879, NA12881, NA12882, NA12885, NA12886) are full siblings,
# grandchildren of two unrelated grandparent couples: NA12889/12890 (paternal)
# and NA12891/12892 (maternal). We expect long shared IBD segments between the
# siblings, and IBD between each sibling and the corresponding grandparent side.
#
# Region: chr12:40,000,001-130,000,000 (90 Mb, the full long q arm of chr12,
# clear of the centromere at ~34-38 Mb and of both telomeres). Window: 10 kb.

set -euo pipefail

mkdir -p solutions

bin/ibd \
    --similarity-file data/identity_chr12_pedigree.tsv \
    --region chr12:40000001-130000000 \
    --size 10000 \
    --population Generic \
    --min-len-bp 2000000 \
    --min-lod 3.0 \
    --baum-welch-iters 20 \
    --output solutions/ibd_segments.tsv

echo
echo "Detected IBD segments (first 40):"
column -t -s $'\t' solutions/ibd_segments.tsv | awk 'NR<=40'
echo
echo "Total segments: $(($(wc -l < solutions/ibd_segments.tsv) - 1))"
