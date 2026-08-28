from pathlib import Path

import pandas as pd


DATA_PATH = Path(__file__).with_name("F2s_7_0.parquet")
FEATURES = ["start_v", "final_v", "initial_g", "final_g"]


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
