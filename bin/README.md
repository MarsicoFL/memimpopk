# Pre-built binaries

Four binaries, all stripped, x86_64 Linux ELF, dynamically linked
against `glibc ≥ 2.31` (Ubuntu 20.04 or newer; any Linux from 2020+).

| binary | size | purpose |
|---|---|---|
| `ibs` | ~1 MB | wrapper around `impg similarity` (not used in this workshop) |
| `ibd` | ~1.3 MB | 2-state HMM for IBD segment detection |
| `ancestry` | ~2.8 MB | N-state HMM for local ancestry inference |
| `jacquard` | ~0.8 MB | kinship coefficient estimation |

Total: ~6 MB.

## Provenance

Built from [`MarsicoFL/IMPOPk`](https://github.com/MarsicoFL/IMPOPk)
with:

```
cargo build --release
strip target/release/{ibs,ibd,ancestry,jacquard}
```

No `target-cpu=native`. No `RUSTFLAGS` other than `-C strip=symbols`.
The build machine had `glibc 2.39`; binaries link only against
`libc.so.6`, `libm.so.6`, and `libgcc_s.so.1`, all of which are
present on every modern Linux.

## Verify

```
$ ./bin/ancestry --help | head -3
Local ancestry inference from pangenome data
...
```

If that fails with a missing-library error, your Linux is older than
2020 and you need to rebuild from source.

## Rebuild

The full Cargo workspace ships locally in [`../source/`](../source/),
so you can rebuild without internet:

```
cd ../source
cargo build --release
cp target/release/{ibs,ibd,ancestry,jacquard} ../bin/
```

Requires Rust 1.70+.
