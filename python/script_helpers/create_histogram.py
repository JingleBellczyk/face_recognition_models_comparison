import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./MAAD_Face.csv")
race_cols = ["Asian", "White", "Black"]

print("Liczba wszystkich wierszy (zdjęć):", len(df))
print("Liczba unikalnych osób (Identity):", df["Identity"].nunique())

def compute_race(subdf):
    # Sprawdź, czy któraś rasa ma wartość 1
    has_1 = any(subdf[r].eq(1).any() for r in race_cols)
    if not has_1:
        return "Undefined"
    # Wybierz rasę z najwyższą średnią wartością
    race_means = {r: subdf[r].mean() for r in race_cols}
    best_race = max(race_means, key=race_means.get)
    return best_race

race_per_identity = (
    df.groupby("Identity")
      .apply(lambda g: compute_race(g))
      .reset_index(name="Race")
)

print("Przykładowe wyniki:\n", race_per_identity.head())

# Liczba osób w każdej rasie
counts = race_per_identity["Race"].value_counts()
print(counts)

# Histogram
plt.figure(figsize=(7,5))
counts.plot(kind="bar", color=["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"])
plt.title("Liczba osób wg rasy")
plt.xlabel("Rasa")
plt.ylabel("Liczba osób")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("race_histogram.png", dpi=300)
plt.show()

# Zapis do CSV
race_per_identity.to_csv("identity_race.csv", index=False)
