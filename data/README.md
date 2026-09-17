# Data requirements

The experimental Parquet files are intentionally excluded from Git. Do not
copy raw data into this repository unless public redistribution has been
explicitly authorised by the data owner.

## Source dataset

Most scripts expect the authorised source file at:

```text
MemristorML/F2s_7_0.parquet
```

This filename is ignored by Git through the broader `*.parquet` rule.

## Required structure

The dataframe must use a four-level Pandas `MultiIndex` in this order:

1. `RESET_pw`
2. `RESET_target`
3. `Device`
4. `Cycle`

The columns must also be a two-level `MultiIndex`. Both the `SET` and `RESET`
column groups must contain at least:

```text
start_v
final_v
initial_g
final_g
has_error
```

Voltage values are measured in volts and conductance values in siemens.

## Pairing rules

For `SET_n -> RESET_n`, inputs and targets are taken from the same dataframe
row. For `RESET_n -> SET_(n+1)`, rows are first grouped by `RESET_pw`,
`RESET_target`, and `Device`; the RESET inputs are shifted by one row only
inside each group. A candidate is retained only when the target cycle equals
the input cycle plus one.

A pair is excluded when any selected input or target is missing, or when a
relevant `has_error` flag is true or missing.

## External evaluation dataset

`test_external_devices.py` accepts the external Parquet path as its first
command-line argument. It expects the same index and column structure. The
script validates the schema, excludes the source device from external metrics,
and reports both pooled and per-device results.

Example:

```bash
python test_external_devices.py /authorised/path/external_devices.parquet
```

External-device identifiers, row-level data, and per-device metrics may still
be sensitive derived information. Keep generated files private until their
publication has been approved.
