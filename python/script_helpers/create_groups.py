import pandas as pd

df = pd.read_csv("./MAAD_Face.csv")

race_cols = ["Asian", "White", "Black"]

def compute_identity_group(subdf):
    male_val = subdf["Male"].mean()
    is_male = male_val > 0

    race_means = {r: subdf[r].mean() for r in race_cols}
    best_race = max(race_means, key=race_means.get)
    race_idx = {"Asian": 0, "Black": 1, "White": 2}[best_race]

    return 3 + race_idx if is_male else race_idx

# ⬇️ Używamy include_groups=False, aby uniknąć ostrzeżenia
groups = (
    df.groupby("Identity", group_keys=False)
      .apply(lambda g: pd.Series({"Group": compute_identity_group(g)}))
      .reset_index()
)

# Zapisz wynik
groups.to_csv("identity_groups.csv", index=False)

print(groups.head())
# /home/agata-ner/Documents/semestr9/glebokie_uczenie/preprocessing/script_helpers/create_groups.py