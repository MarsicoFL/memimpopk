# Source code

Cargo workspace with the four CLIs shipped in `../bin/`. Build it
yourself if the pre-built binaries don't run on your system or if you
want to inspect / modify the algorithms.

## Requirements

- Rust 1.70 or newer ([rustup.rs](https://rustup.rs/))
- A C linker (`build-essential` on Debian/Ubuntu, Xcode CLT on macOS)

No system libraries beyond `glibc`/`libc`, `libm`, `libgcc_s`.

## Build

```
cd workshop/source
cargo build --release
```

The four binaries land in `target/release/`:

```
target/release/ibs
target/release/ibd
target/release/ancestry
target/release/jacquard
```

The release profile (`Cargo.toml`) already enables `lto = true`,
`codegen-units = 1`, `strip = true`, and `opt-level = 3`. You do **not**
need to pass `RUSTFLAGS` or run `strip` manually.

## Replace the pre-built binaries

```
cp target/release/{ibs,ibd,ancestry,jacquard} ../bin/
```

## Provenance

Snapshot of the `MarsicoFL/IMPOPk` workspace (workspace version
`0.2.0`). Build flags match `bin/README.md`: no `target-cpu=native`,
no extra `RUSTFLAGS`. Binaries are dynamically linked against
`glibc ≥ 2.31` (Ubuntu 20.04 +).

## Layout

```
Cargo.toml             # workspace manifest
Cargo.lock             # pinned dependency versions
src/
├── common/            # shared types (Region, Window, error)
├── ibs-cli/           # `ibs`        wrapper around `impg similarity`
├── ibd-cli/           # `ibd`        2-state HMM, IBD detection
├── ancestry-cli/      # `ancestry`   N-state HMM, local ancestry
└── jacquard-cli/      # `jacquard`   kinship coefficients
```

## Tests

```
cargo test --workspace --release
```

## License

MIT (same as the rest of the workshop).
