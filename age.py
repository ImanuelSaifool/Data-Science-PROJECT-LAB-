print ("safwan has a cheating kink")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------------------------------
# dataset's entrance into this world
df2021p1 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2021_data_part1.csv")
df2021p2 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2021_data_part2.csv")
df2022 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2022%20data.csv")
df2023 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2023%20data.csv")


# renaming stuff
    # out of pocket cost between different years
df2021p1 = df2021p1.rename(columns={"TOTSLF21": "TOTSLF"})
df2021p2 = df2021p2.rename(columns={"TOTSLF21": "TOTSLF"}) 
df2022 = df2022.rename(columns={"TOTSLF22": "TOTSLF"})
df2023 = df2023.rename(columns={"TOTSLF23": "TOTSLF"})
    # family income between different years
df2021p1 = df2021p1.rename(columns={"FAMINC21": "FAMINC"})
df2021p2 = df2021p2.rename(columns={"FAMINC21": "FAMINC"}) 
df2022 = df2022.rename(columns={"FAMINC22": "FAMINC"})
df2023 = df2023.rename(columns={"FAMINC23": "FAMINC"})
    # insurance covered between different years
df2021p1 = df2021p1.rename(columns={"INSCOV21": "INSCOV"})
df2021p2 = df2021p2.rename(columns={"INSCOV21": "INSCOV"}) 
df2022 = df2022.rename(columns={"INSCOV22": "INSCOV"})
df2023 = df2023.rename(columns={"INSCOV23": "INSCOV"})

# combining them
main_df = pd.concat([df2021p1, df2021p2, df2022, df2023], axis=0)


# data filter
features = ["FAMINC","TOTSLF", "AGELAST", "SEX", "CABLADDR", "CABREAST", "CACERVIX", "CACOLON", "CALUNG", "CALYMPH", "CAMELANO", "CAOTHER"]
clean_df = main_df[main_df['CANCERDX'] == 1].copy() 
clean_df = clean_df[(clean_df[features] >= 0).all(axis=1)]

def is_unable(row):
    # Change from physical inability to FINANCIAL inability
    val = row.get('PYUNBL42')
    if val == 1: return 1 # Yes, unable to pay
    elif val == 2: return 0 # No, was able to pay
    else: return np.nan

clean_df['UNABLE'] = clean_df.apply(is_unable, axis=1)

clean_df = clean_df.dropna(subset=['UNABLE'])

# we need to define the policy groups for our study
def get_policy_group(code):
    if code == 1: return "Private (Market)"
    elif code == 2: return "Public (Subsidized)"
    elif code == 3: return "Uninsured"
    else: return "Unknown"

clean_df['POLICY_GROUP'] = clean_df['INSCOV'].apply(get_policy_group)
# ----------------------------------------------------------------------------------------------------------------------------------------------
# ==========================================
# FIRST: Summary Statistics
# ==========================================
summary_stats = clean_df[features].describe()
print(summary_stats)
print(clean_df.groupby('UNABLE')['TOTSLF'].mean())

# ----------------------------------------------------------------------------------------------------------------------------------------------
# White grid for plottsss
sns.set_style("whitegrid")

# ==========================================
# 4. RESULTS FOR PROPOSAL
# ==========================================
print("--- SUMMARY STATISTICS (For Proposal) ---")
print(clean_df[["TOTSLF", "FAMINC", "UNABLE"]].describe())

print("\n--- THE SUBSIDY SIGNAL: Quitting Rates by Group ---")
# This table proves if Public Insurance is working better than Uninsured
policy_stats = clean_df.groupby('POLICY_GROUP')[['UNABLE', 'TOTSLF', 'FAMINC']].mean()
print(policy_stats)

# ----------------------------------------------------
# Select relevant variables
# Replace:
#   AGEVAR       → MEPS age variable
#   QUIT_CANCER  → your quitting-treatment variable
# ----------------------------------------------------
clean_df["Age Group"] = pd.cut( 
    clean_df["AGELAST"], 
    bins=[0, 17, 34, 49, 64, 120], 
    labels=["0–17", "18–34", "35–49", "50–64", "65+"] 
    )
clean_df['Percentage Unable to Afford Cancer Treatment'] = clean_df.apply(is_unable, axis=1)

from matplotlib.ticker import PercentFormatter

plt.figure(figsize=(8,5)) 
sns.barplot(data=clean_df, x="Age Group", y="Percentage Unable to Afford Cancer Treatment", estimator="mean", errorbar=None) 
plt.title("Inability to Afford Cancer Treatment by Age Group") 
plt.ylabel("Percentage Unable to Afford Cancer Treatment") 
plt.xlabel("Age Group") 
plt.ylim(0,1) 
plt.gca().yaxis.set_major_formatter(PercentFormatter(1)) 
plt.show()
