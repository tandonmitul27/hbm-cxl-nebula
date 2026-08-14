# Validation

<!-- @tandonmitul27 -- authored document. -->

What `make check` compares, and against what. It runs every comparison the
model rests on and exits nonzero if any drifts. All references are external —
a published standard, a vendor datasheet, or a peer-reviewed measurement.

---

## The checks

| Check | Ours | Reference |
|---|---|---|
| HBM3 stack bandwidth | 707.8 GB/s | 670 GB/s implied by the H100 datasheet |
| HBM3E stack bandwidth | 1027.3 GB/s | 84 % of the 1228.8 spec peak |
| CXL ASIC added latency | 159.7 ns | 154 ns, real CXL ASIC |
| CXL FPGA added latency | 248.5 ns | 245 ns, real CXL FPGA |
| CXL 3.0 x16 bandwidth | 105.6 GB/s | 87 % of 121 spec |
| Mixtral expert over PCIe4 | 15.0 ms | FloE, ~15 ms |
| HBM3 / HBM3E energy | 3.97 pJ/bit | O'Connor, MICRO'17 |
| DDR5 energy | 11.8 pJ/bit | DDR/GDDR device class |
| Row-miss penalty | 33.2 / 34.2 ns | configured tRP + tRCD |
| Read/write mix | 41.9 GB/s | 95 % of read-only throughput |
| Address map, 4 models | ≤ 0.3 % error | published checkpoint sizes |

A check that cannot fail is not a check: tolerances are tight enough that a
real regression trips them.

---

## Running it

```bash
make check         # everything, including the long bandwidth sweeps
make check-fast    # skips the sweeps, ~2 min
```

Run `check-fast` after any edit and the full suite before trusting a result.
A failure prints the measured value, the reference, and the tolerance that
was exceeded.

---

## What the checks do not cover

Two categories are validated by construction rather than by comparison, and
both are labelled where they appear:

* **HBM energy** matches its anchor *because it was calibrated to*. The check
  confirms the calibration still holds after a config change; it is not
  independent evidence. See
  [why HBM energy is calibrated](PARAMETERS.md#energy).
* **CXL link energy** has no simulator support and no vendor split to compare
  against, so it is carried as a swept bracket rather than a value.

The estimates that remain, and what would be needed to retire each, are listed
at the end of [CALIBRATION.md](CALIBRATION.md#what-remains-an-estimate).
