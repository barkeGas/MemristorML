# Generated results

This directory contains versioned outputs from the analysis scripts. They make
the reported model-selection, parameter-matrix, and external-evaluation results
inspectable without distributing the raw experimental data.

## Ridge comparison

Running `python ridge_regression.py` creates:

- `set_to_reset_ridge_alpha_search.csv`
- `set_to_reset_ridge_comparison.csv`
- `reset_to_next_set_ridge_alpha_search.csv`
- `reset_to_next_set_ridge_comparison.csv`

The alpha-search tables contain validation R-squared values for every target
and their mean. The comparison tables contain Linear and Ridge test metrics.

## Parameter matrices

Running `python show_parameter_matrices.py` creates standardized-input and
original-input augmented parameter matrices for both prediction tasks. Each
matrix has five rows (one bias row and four input-weight rows) and four target
columns.

## External-device evaluation

Running `python test_external_devices.py PATH` creates:

- `external_device_alphas.csv`
- `external_device_diagnostics.csv`
- `external_device_distribution_shift.csv`
- `external_device_summary.csv`
- `external_device_per_device.csv`

The summary contains pooled and device-macro metrics. The per-device file is
the most detailed and includes experimental device identifiers and row counts.

All generated results should be interpreted together with the exact source
commit, dataset version, experimental conditions, and validation procedure.
