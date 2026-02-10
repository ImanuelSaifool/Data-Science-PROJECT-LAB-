print("safwan has a cheating kink")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import PercentFormatter

# ----------------------------------------------------------------------------------------------------------------------------------------------
# dataset's entrance into this world
df2021p1 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2021_data_part1.csv")
df2021p2 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2021_data_part2.csv")
df2022 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2022%20data.csv")
df2023 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2023%20data.csv")

# renaming stuff
df2021p1 = df2021p1.rename(columns={"TOTSLF21": "TOTSLF", "FAMINC21": "FAMINC", "INSCOV21": "INSCOV"})
df2021p2 = df2021p2.rename(columns={"TOTSLF21": "TOTSLF", "FAMINC21": "FAMINC", "INSCOV21": "INSCOV"})
df2022 = df2022.rename(columns={"TOTSLF22": "TOTSLF", "FAMINC22": "FAMINC", "INSCOV22": "INSCOV"})
df2023 = df2023.rename(columns={"TOTSLF23": "TOTSLF", "FAMINC23": "FAMINC", "INSCOV23": "INSCOV"})

# combining them
main_df = pd.concat([df2021p1, df2021p2, df2022, df2023], axis=0)

# ----------------------------------------------------------------------------------------------------------------------------------------------
# FILTERING (fixed)
clean_df = main_df[main_df["CANCERDX"] == 1].copy()

def is_unable(row):
    val = row.get('PYUNBL42')
    if val == 1: return 1
    elif val == 2: return 0
    else: return np.nan

clean_df['UNABLE'] = clean_df.apply(is_unable, axis=1)

clean_df = clean_df.dropna(subset=['UNABLE'])

clean_df = clean_df[
    (clean_df["SEX"] >= 0) &
    (clean_df["UNABLE"] >= 0)
]

# ----------------------------------------------------------------------------------------------------------------------------------------------
# VISUALIZATION
sns.set_style("whitegrid")

sex_map = {1: "Male", 2: "Female"}
clean_df["Assigned Sex"] = clean_df["SEX"].map(sex_map)

clean_df["UNABLE"] = clean_df["UNABLE"].astype(float)

plt.figure(figsize=(6,4))
ax = sns.barplot(
    data=clean_df,
    x="Assigned Sex",
    y="UNABLE",
    estimator="mean",
    errorbar=None
)
ax.set_title("Inability to Afford Cancer Treatment by Assigned Sex")
ax.set_ylabel("Percentage Unable to Afford Cancer Treatment")
ax.set_xlabel("Assigned Sex")
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(PercentFormatter(1))
plt.tight_layout()
plt.show()
