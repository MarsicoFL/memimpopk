#!/usr/bin/env bash
# 5-second sanity check: does this machine have everything needed to run
# the workshop tutorial? Prints OK on success, the failing component on
# error. Safe to run multiple times.
set -u

WS="$(cd "$(dirname "$0")" && pwd)"
cd "$WS"

fail() {
    printf '\033[31mFAIL\033[0m  %s\n' "$1" >&2
    exit 1
}

# 1. binaries present and executable
for b in ancestry ibd ibs jacquard; do
    [[ -x "bin/$b" ]] || fail "bin/$b missing or not executable"
done

# 2. the binaries actually run on this machine (catches glibc / ARCH mismatch)
if ! ./bin/ancestry --help >/dev/null 2>&1; then
    cat >&2 <<EOF
FAIL  bin/ancestry could not execute on this system.

The shipped binaries are Linux x86_64 (glibc >= 2.31). If you are on
macOS, ARM, or older Linux, build from the bundled source (no internet
required):

  cd $WS/source
  cargo build --release
  cp target/release/{ancestry,ibd,ibs,jacquard} $WS/bin/

Or clone the upstream:

  git clone https://github.com/MarsicoFL/impop
  cd impop
  cargo build --release
  cp target/release/{ancestry,ibd,ibs,jacquard} $WS/bin/

then re-run this script.
EOF
    exit 1
fi

# 3. Python 3 with numpy and matplotlib
if ! command -v python3 >/dev/null 2>&1; then
    fail "python3 not found in PATH"
fi
if ! python3 -c "import numpy, matplotlib" >/dev/null 2>&1; then
    fail "Python is present but missing numpy and/or matplotlib. Install with: pip install numpy matplotlib"
fi

# 4. data files present and non-empty
for f in data/identity_chr12_pedigree.tsv \
         data/identity_chr12_admix.tsv \
         data/populations.tsv \
         data/queries.txt \
         data/ground_truth_tracts.tsv \
         data/pedigree_populations.tsv \
         data/pedigree_queries.txt; do
    [[ -s "$f" ]] || fail "$f missing or empty"
done

# 5. all student-facing scripts are executable
for s in code/01_explore_ibs.py \
         code/02_run_ibd.sh \
         code/03_run_ancestry.sh \
         code/04_plot_painting.py \
         code/05_paint_pedigree.sh \
         code/06_plot_pedigree_painting.py \
         code/07_plot_posteriors.py; do
    [[ -x "$s" ]] || fail "$s missing or not executable (try: chmod +x $s)"
done

printf '\033[32mOK\033[0m  workshop ready (binaries run, Python deps present, data + scripts in place).\n'
printf '    open tutorial/tutorial.pdf and follow from Section 0.\n'
