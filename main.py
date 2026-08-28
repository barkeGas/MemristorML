from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(__file__).with_name("F2s_7_0.parquet")
PLOTS_DIRECTORY = Path(__file__).with_name("plots")
FEATURES = ["start_v", "final_v", "initial_g", "final_g"]
TARGET_UNITS = {
    "start_v": "V",
    "final_v": "V",
    "initial_g": "S",
    "final_g": "S",
}


# 1. Ucitavanje celog dataseta
data = pd.read_parquet(DATA_PATH)

print("=== DATASET ===")
print(f"Fajl: {DATA_PATH.name}")
print(f"Broj redova: {len(data)}")
print(f"Broj kolona: {data.shape[1]}")


# 2. X sadrzi SET metrike, a Y odgovarajuce RESET metrike.
# Indeks se zadrzava da bi svaki SET ostao poravnat sa RESET-om istog ciklusa.
X = data.loc[:, [("SET", feature) for feature in FEATURES]].copy()
Y = data.loc[:, [("RESET", feature) for feature in FEATURES]].copy()

# Uklanjamo prvi nivo naziva kolona jer je vec jasno da je X=SET, a Y=RESET.
X.columns = FEATURES
Y.columns = FEATURES


# 3. Pregled NaN vrednosti samo u osam metrika koje ce model koristiti.
print("\n=== NaN PREGLED PRE FILTRIRANJA ===")
print("X (SET):")
print(X.isna().sum())
print("\nY (RESET):")
print(Y.isna().sum())


# 4. Pregled SET i RESET has_error oznaka.
error_flags = data.loc[
    :, [("SET", "has_error"), ("RESET", "has_error")]
].copy()
error_flags.columns = ["SET_has_error", "RESET_has_error"]

print("\n=== has_error PREGLED ===")
for column in error_flags.columns:
    print(f"{column}=True: {int(error_flags[column].eq(True).sum())}")
    print(f"{column}=NaN:  {int(error_flags[column].isna().sum())}")


# 5. Red je validan samo ako svih osam metrika postoji i nijedna
# operacija nema gresku. Nedostajuca has_error oznaka se tretira kao greska.
has_nan = X.isna().any(axis=1) | Y.isna().any(axis=1)
has_error = error_flags.fillna(True).astype(bool).any(axis=1)
valid_rows = ~(has_nan | has_error)

print("\n=== FILTRIRANJE ===")
print(f"Redovi sa NaN u X ili Y: {int(has_nan.sum())}")
print(f"Redovi sa SET/RESET greskom: {int(has_error.sum())}")
print(f"Ukupno izbaceno redova: {int((~valid_rows).sum())}")

X = X.loc[valid_rows].copy()
Y = Y.loc[valid_rows].copy()


# 6. Zavrsna provera pre bilo kakvog treninga.
print("\n=== KONACNI X I Y ===")
print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")
print(f"X i Y imaju isti indeks: {X.index.equals(Y.index)}")
print(f"Preostali NaN u X: {int(X.isna().sum().sum())}")
print(f"Preostali NaN u Y: {int(Y.isna().sum().sum())}")

print("\nPrvih 5 redova X (SET):")
print(X.head())

print("\nPrvih 5 redova Y (RESET):")
print(Y.head())


# 7. Hronoloska podela: raniji ciklusi sluze za trening, a kasniji
# za validaciju i zavrsni test. Redosled se ne mesa (nema shuffle-a).
cycle_numbers = X.index.get_level_values("Cycle")
if not cycle_numbers.is_monotonic_increasing:
    raise ValueError("Ciklusi nisu hronoloski sortirani.")

train_end = int(len(X) * 0.80)
validation_end = int(len(X) * 0.90)

X_train = X.iloc[:train_end].copy()
Y_train = Y.iloc[:train_end].copy()

X_validation = X.iloc[train_end:validation_end].copy()
Y_validation = Y.iloc[train_end:validation_end].copy()

X_test = X.iloc[validation_end:].copy()
Y_test = Y.iloc[validation_end:].copy()


def cycle_range(frame):
    """Vraca prvi i poslednji broj ciklusa u jednom delu dataseta."""
    cycles = frame.index.get_level_values("Cycle")
    return f"{cycles[0]}-{cycles[-1]}"


print("\n=== HRONOLOSKA PODELA 80/10/10 ===")
print(f"Train:      X{X_train.shape}, Y{Y_train.shape}, ciklusi {cycle_range(X_train)}")
print(
    f"Validation: X{X_validation.shape}, Y{Y_validation.shape}, "
    f"ciklusi {cycle_range(X_validation)}"
)
print(f"Test:       X{X_test.shape}, Y{Y_test.shape}, ciklusi {cycle_range(X_test)}")

print("\n=== PROVERA PODELE ===")
print(
    "Ukupan broj redova je sacuvan: "
    f"{len(X_train) + len(X_validation) + len(X_test) == len(X)}"
)
print(f"Train X/Y indeks poravnat: {X_train.index.equals(Y_train.index)}")
print(
    "Validation X/Y indeks poravnat: "
    f"{X_validation.index.equals(Y_validation.index)}"
)
print(f"Test X/Y indeks poravnat: {X_test.index.equals(Y_test.index)}")


# 8. Multi-output linear regression baseline.
# StandardScaler se fituje samo na X_train unutar pipeline-a.
linear_model = make_pipeline(StandardScaler(), LinearRegression())
linear_model.fit(X_train, Y_train)

Y_validation_predicted = pd.DataFrame(
    linear_model.predict(X_validation),
    index=Y_validation.index,
    columns=FEATURES,
)
Y_test_predicted = pd.DataFrame(
    linear_model.predict(X_test),
    index=Y_test.index,
    columns=FEATURES,
)


def regression_metrics(y_true, y_predicted):
    """Racuna metrike zasebno za svaki RESET target."""
    rows = []

    for target in FEATURES:
        mse = mean_squared_error(y_true[target], y_predicted[target])
        rows.append(
            {
                "target": target,
                "MAE": mean_absolute_error(
                    y_true[target], y_predicted[target]
                ),
                "RMSE": np.sqrt(mse),
                "R2": r2_score(y_true[target], y_predicted[target]),
            }
        )

    return pd.DataFrame(rows).set_index("target")


validation_metrics = regression_metrics(
    Y_validation, Y_validation_predicted
)
test_metrics = regression_metrics(Y_test, Y_test_predicted)

print("\n=== LINEAR REGRESSION: VALIDATION METRIKE ===")
print(validation_metrics.to_string(float_format=lambda value: f"{value:.6g}"))

print("\n=== LINEAR REGRESSION: TEST METRIKE ===")
print(test_metrics.to_string(float_format=lambda value: f"{value:.6g}"))


# 9. Pet ravnomerno rasporedjenih primera iz hronoloskog test skupa.
sample_positions = np.linspace(
    0, len(X_test) - 1, num=5, dtype=int
)

print("\n=== POJEDINACNI TEST PRIMERI ===")
for position in sample_positions:
    cycle = X_test.iloc[[position]].index.get_level_values("Cycle")[0]
    comparison = pd.DataFrame(
        {
            "stvarni_RESET": Y_test.iloc[position],
            "predvidjeni_RESET": Y_test_predicted.iloc[position],
        }
    )
    comparison["greska"] = (
        comparison["predvidjeni_RESET"] - comparison["stvarni_RESET"]
    )
    comparison["apsolutna_greska"] = comparison["greska"].abs()

    print(f"\n--- Ciklus {cycle} ---")
    print("SET input:")
    print(X_test.iloc[position].to_string(float_format=lambda value: f"{value:.6g}"))
    print("RESET: stvarno, predvidjeno i odstupanje:")
    print(comparison.to_string(float_format=lambda value: f"{value:.6g}"))


# 10. Measured-vs-predicted grafik za ceo test skup.
PLOTS_DIRECTORY.mkdir(exist_ok=True)
plot_path = PLOTS_DIRECTORY / "linear_regression_test_predictions.png"

figure, axes = plt.subplots(2, 2, figsize=(11, 9))
axes = axes.flatten()
sample_cycles = X_test.iloc[sample_positions].index.get_level_values("Cycle")

for axis, target in zip(axes, FEATURES):
    actual = Y_test[target]
    predicted = Y_test_predicted[target]
    lower = min(actual.min(), predicted.min())
    upper = max(actual.max(), predicted.max())
    padding = (upper - lower) * 0.05

    axis.scatter(
        actual,
        predicted,
        alpha=0.35,
        s=18,
        label="Svi test ciklusi",
    )
    axis.plot(
        [lower - padding, upper + padding],
        [lower - padding, upper + padding],
        linestyle="--",
        color="black",
        linewidth=1.2,
        label="Idealno: y = x",
    )
    axis.scatter(
        actual.iloc[sample_positions],
        predicted.iloc[sample_positions],
        s=42,
        marker="x",
        linewidths=1.5,
        label="Izdvojeni primeri",
    )

    for sample_position, cycle in zip(sample_positions, sample_cycles):
        axis.annotate(
            str(cycle),
            (
                actual.iloc[sample_position],
                predicted.iloc[sample_position],
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    unit = TARGET_UNITS[target]
    axis.set_title(f"RESET {target}")
    axis.set_xlabel(f"Stvarna vrednost [{unit}]")
    axis.set_ylabel(f"Predvidjena vrednost [{unit}]")
    axis.grid(alpha=0.25)
    axis.text(
        0.04,
        0.96,
        (
            f"MAE = {test_metrics.loc[target, 'MAE']:.4g} {unit}\n"
            f"R2 = {test_metrics.loc[target, 'R2']:.3f}"
        ),
        transform=axis.transAxes,
        verticalalignment="top",
    )

axes[0].legend(fontsize=8)
figure.suptitle(
    "Linearna regresija: stvarni i predvidjeni RESET na test skupu",
    fontsize=14,
)
figure.tight_layout()
figure.savefig(plot_path, dpi=160, bbox_inches="tight")

print(f"\nGrafik je sacuvan u: {plot_path}")
plt.show()
