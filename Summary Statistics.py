import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn
from sklearn.feature_selection import mutual_info_classif

# ----------------------------------------------------------------------------------------------------------------------------------------------
# Data's entrance into this teeny weeny world
# ----------------------------------------------------------------------------------------------------------------------------------------------
df2019 = pd.read_csv("https://github.com/ImanuelSaifool/Data-Science-PROJECT-LAB-/raw/main/OneDrive/Desktop/Coding%20Projects/h216.csv")
df2020 = pd.read_csv("https://github.com/ImanuelSaifool/Data-Science-PROJECT-LAB-/raw/main/OneDrive/Desktop/Coding%20Projects/H224.csv")
df2021p1 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2021_data_part1.csv")
df2021p2 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2021_data_part2.csv")
df2022 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2022%20data.csv")
df2023 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2023%20data.csv")

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 2. Standardizing column names
    # we do this so that we can easily integrate multiple data files without changing the name on the raw data
# ----------------------------------------------------------------------------------------------------------------------------------------------

# Treating inflation
    # Family income
df2019["FAMINC19"] = df2019["FAMINC19"] * 1.19
df2020["FAMINC20"] = df2020["FAMINC20"] * 1.17
df2021p1['FAMINC21'] = df2021p1['FAMINC21'] * 1.12
df2021p2['FAMINC21'] = df2021p2['FAMINC21'] * 1.12
df2022['FAMINC22'] = df2022['FAMINC22'] * 1.04

df2019["TOTSLF19"] = df2019["TOTSLF19"] * 1.19
df2020["TOTSLF20"] = df2020["TOTSLF20"] * 1.17
df2021p1['TOTSLF21'] = df2021p1['TOTSLF21'] * 1.12
df2021p2['TOTSLF21'] = df2021p2['TOTSLF21'] * 1.12
df2022['TOTSLF22'] = df2022['TOTSLF22'] * 1.04

# Out of pocket cost
df2019 = df2019.rename(columns={"TOTSLF19": "TOTSLF"})
df2020 = df2020.rename(columns={"TOTSLF20": "TOTSLF"})
df2021p1 = df2021p1.rename(columns={"TOTSLF21": "TOTSLF"})
df2021p2 = df2021p2.rename(columns={"TOTSLF21": "TOTSLF"})
df2022 = df2022.rename(columns={"TOTSLF22": "TOTSLF"})
df2023 = df2023.rename(columns={"TOTSLF23": "TOTSLF"})

# Family income
df2019 = df2019.rename(columns={"FAMINC19": "FAMINC"})
df2020 = df2020.rename(columns={"FAMINC20": "FAMINC"})
df2021p1 = df2021p1.rename(columns={"FAMINC21": "FAMINC"})
df2021p2 = df2021p2.rename(columns={"FAMINC21": "FAMINC"})
df2022 = df2022.rename(columns={"FAMINC22": "FAMINC"})
df2023 = df2023.rename(columns={"FAMINC23": "FAMINC"})

# Insurance covered
df2019 = df2019.rename(columns={"INSCOV19": "INSCOV"})
df2020 = df2020.rename(columns={"INSCOV20": "INSCOV"})
df2021p1 = df2021p1.rename(columns={"INSCOV21": "INSCOV"})
df2021p2 = df2021p2.rename(columns={"INSCOV21": "INSCOV"})
df2022 = df2022.rename(columns={"INSCOV22": "INSCOV"})
df2023 = df2023.rename(columns={"INSCOV23": "INSCOV"})

# Renaming Medicare
df2019 = df2019.rename(columns={"TOTMCR19": "TOTMCR"})
df2020 = df2020.rename(columns={"TOTMCR20": "TOTMCR"})
df2021p1 = df2021p1.rename(columns={"TOTMCR21": "TOTMCR"})
df2021p2 = df2021p2.rename(columns={"TOTMCR21": "TOTMCR"})
df2022 = df2022.rename(columns={"TOTMCR22": "TOTMCR"})
df2023 = df2023.rename(columns={"TOTMCR23": "TOTMCR"})

# Renaming Medicaid
df2019 = df2019.rename(columns={"TOTMCD19": "TOTMCD"})
df2020 = df2020.rename(columns={"TOTMCD20": "TOTMCD"})
df2021p1 = df2021p1.rename(columns={"TOTMCD21": "TOTMCD"})
df2021p2 = df2021p2.rename(columns={"TOTMCD21": "TOTMCD"})
df2022 = df2022.rename(columns={"TOTMCD22": "TOTMCD"})
df2023 = df2023.rename(columns={"TOTMCD23": "TOTMCD"})

# Renaming Veterans Affair
df2019 = df2019.rename(columns={"TOTVA19": "TOTVA"})
df2020 = df2020.rename(columns={"TOTVA20": "TOTVA"})
df2021p1 = df2021p1.rename(columns={"TOTVA21": "TOTVA"})
df2021p2 = df2021p2.rename(columns={"TOTVA21": "TOTVA"})
df2022 = df2022.rename(columns={"TOTVA22": "TOTVA"})
df2023 = df2023.rename(columns={"TOTVA23": "TOTVA"})

#Renaming Other Federal
df2019 = df2019.rename(columns={"TOTOFD19": "TOTOFD"})
df2020 = df2020.rename(columns={"TOTOFD20": "TOTOFD"})
df2021p1 = df2021p1.rename(columns={"TOTOFD21": "TOTOFD"})
df2021p2 = df2021p2.rename(columns={"TOTOFD21": "TOTOFD"})
df2022 = df2022.rename(columns={"TOTOFD22": "TOTOFD"})
df2023 = df2023.rename(columns={"TOTOFD23": "TOTOFD"})

#Renaming State
df2019 = df2019.rename(columns={"TOTSTL19": "TOTSTL"})
df2020 = df2020.rename(columns={"TOTSTL20": "TOTSTL"})
df2021p1 = df2021p1.rename(columns={"TOTSTL21": "TOTSTL"})
df2021p2 = df2021p2.rename(columns={"TOTSTL21": "TOTSTL"})
df2022 = df2022.rename(columns={"TOTSTL22": "TOTSTL"})
df2023 = df2023.rename(columns={"TOTSTL23": "TOTSTL"})

#Renaming Worker's Comp
df2019 = df2019.rename(columns={"TOTWCP19": "TOTWCP"})
df2020 = df2020.rename(columns={"TOTWCP20": "TOTWCP"})
df2021p1 = df2021p1.rename(columns={"TOTWCP21": "TOTWCP"})
df2021p2 = df2021p2.rename(columns={"TOTWCP21": "TOTWCP"})
df2022 = df2022.rename(columns={"TOTWCP22": "TOTWCP"})
df2023 = df2023.rename(columns={"TOTWCP23": "TOTWCP"})


# Combining datasets
main_df = pd.concat([df2019, df2020, df2021p1, df2021p2, df2022, df2023], axis=0)

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 3. FILTERING & CLEANING
# ----------------------------------------------------------------------------------------------------------------------------------------------
# Define feature lists
demog_features = ["FAMINC", "TOTSLF", "AGELAST", "SEX", "DLAYCA42", "AFRDCA42", "DLAYPM42", "AFRDPM42"]
cancer_features = ["CABLADDR", "CABREAST", "CACERVIX", "CACOLON", "CALUNG", "CALYMPH", "CAMELANO", "CAOTHER", "CAPROSTA", "CASKINNM", "CASKINDK", "CAUTERUS"]
other_disease_features = ["DIABDX_M18", "HIBPDX", "CHDDX", "ANGIDX", "MIDX", "OHRTDX", "STRKDX", "CHOLDX", "EMPHDX", "ASTHDX", "CHBRON31", "ARTHDX"]
insurance_features = ["TOTMCR", "TOTMCD", "TOTVA", "TOTOFD", "TOTSTL", "TOTWCP"]
features = demog_features + cancer_features + other_disease_features

cancer_map = {
    "CABLADDR": "Bladder Cancer",
    "CABREAST": "Breast Cancer",
    "CACERVIX": "Cervix Cancer",
    "CACOLON": "Colon Cancer",
    "CALUNG": "Lung Cancer",
    "CALYMPH": "Lymph Cancer",
    "CAMELANO": "Melano Cancer",
    "CAOTHER": "Other Cancer",
    "CAPROSTA": "Prostate Cancer",
    "CASKINNM": "Skin Cancer 1",
    "CASKINDK": "SKin Cancer 2",
    "CAUTERUS": "Uterus Cancer"
}

disease_map = {
    "HIBPDX": "High Blood Pressure",
    "ARTHDX": "Arthritis",
    "CHOLDX": "High Cholesterol",
    "OHRTDX": "Other Heart Disease",
    "DIABDX_M18": "Diabetes",
    "ASTHDX": "Asthma",
    "CHDDX": "Coronary Heart Disease",
    "STRKDX": "Stroke",
    "MIDX": "Heart Attack",
    "EMPHDX": "Emphysema",
    "ANGIDX": "Angina",
    "CHBRON31": "Chronic Bronchitis"
}


# Filter for positive cancer diagnosis and public health insurance
clean_df = main_df[(main_df['CANCERDX'] == 1) & (main_df['INSCOV'] == 2)].copy()

# Dropping duplicates for same person
clean_df = clean_df.drop_duplicates(subset=['DUPERSID'], keep='first')

# Filter negative values for demographics only to prevent logic error
clean_df = clean_df[(clean_df[demog_features] >= 0).all(axis=1)]

# Filter negative values for demographics only to prevent logic error
clean_df[cancer_features] = clean_df[cancer_features].replace([-1,-7, -8, -9], 2)

#categorical encoding
    # for other stuff
def is_unable(row):
    val = row.get('DLAYPM42')
    if val == 1: return 1 # Yes, unable to pay
    elif val == 2: return 0 # No, was able to pay
    else: return np.nan

clean_df['UNABLE'] = clean_df.apply(is_unable, axis=1)
clean_df = clean_df.dropna(subset=['UNABLE'])

clean_df['PUBLIC_TOTAL'] = clean_df[insurance_features].sum(axis=1)

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 4. SUMMARY STATISTICS
# ----------------------------------------------------------------------------------------------------------------------------------------------

print("--- GENERAL SUMMARY STATISTICS (General) ---")
print(clean_df[features].describe()) 

print("\n--- Average Out-of-Pocket Cost by Ability to Pay ---")
print(clean_df.groupby('UNABLE')['TOTSLF'].mean())

print("\n--- Average Family Income ---")
print(clean_df.groupby('UNABLE')['FAMINC'].mean())

print("\n--- Average Public Insurance Coverage ---")
print(clean_df.groupby('UNABLE')['PUBLIC_TOTAL'].mean())

summary_list = []
total_patients = len(clean_df)

for col, name in disease_map.items():
    if col in clean_df.columns:
        # 1. Filter for people with this disease
        disease_subgroup = clean_df[clean_df[col] == 1]
        
        # 2. Calculate Stats
        num_patients = len(disease_subgroup)
        percent = (num_patients / total_patients) * 100
        
        # Calculate AVERAGE cost/income for this group (Not the whole list)
        avg_oop = disease_subgroup['TOTSLF'].mean()
        avg_income = disease_subgroup['FAMINC'].mean()
        avg_public = disease_subgroup['PUBLIC_TOTAL'].mean()

        summary_list.append({
            "Comorbidity": name,
            "Count (N)": num_patients,
            "Prevalence (%)": round(percent, 2),
            "Avg OOP Cost ($)": round(avg_oop, 2),
            "Avg Family Income ($)": round(avg_income, 2),
            "Avg Public Pay ($)": round(avg_public, 2)
        })
# ----------------------------------------------------------------------------------------------------------------------------------------------
# 5. VISUALIZATIONS (Heatmap & Policy Graphs)
# ----------------------------------------------------------------------------------------------------------------------------------------------
X = clean_df[['TOTSLF', 'FAMINC', 'PUBLIC_TOTAL', 'AGELAST']]
y = clean_df['UNABLE']

mi_scores = mutual_info_classif(X, y)

for feature, score in zip(X.columns, mi_scores):
    print(f"{feature}: {score}")

# A. FEATURE CORRELATION HEATMAP (Targeted)
# We select only the "drivers" of adherence to see the signal clearly
plt.figure(figsize=(10, 8))
corr_features = ['UNABLE', 'TOTSLF', 'FAMINC', 'PUBLIC_TOTAL', 'AGELAST']
corr_data = clean_df[corr_features].corr(method='spearman')

sns.heatmap(
    corr_data, 
    annot=True, 
    cmap='coolwarm', 
    fmt=".2f", 
    linewidths=0.5,
    vmin=-1, vmax=1
)
plt.title("Correlation Heatmap: Drivers of Non-Adherence", fontsize=14)
plt.show()

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 5. ADVANCED VISUALIZATIONS
# ----------------------------------------------------------------------------------------------------------------------------------------------
plt.rcParams['figure.figsize'] = (12, 6)

# --- VISUALIZATION 1: The "Financial Squeeze" (Income vs. Costs) ---
# We use boxplots to compare the distributions.
# Note: We use 'showfliers=False' because MEPS has massive outliers that ruin the scale.

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Out-of-Pocket Costs
sns.boxplot(data=clean_df, x='UNABLE', y='TOTSLF', ax=axes[0], 
            showfliers=False, palette="Reds")
axes[0].set_title("Impact of Out-of-Pocket Costs on Adherence", fontsize=14)
axes[0].set_xticklabels(['Able to Pay', 'Financial Toxicity (Unable/Delay)'])
axes[0].set_ylabel("Total Out-of-Pocket Cost ($)")

# Plot 2: Family Income
sns.boxplot(data=clean_df, x='UNABLE', y='FAMINC', ax=axes[1], 
            showfliers=False, palette="Greens")
axes[1].set_title("Protective Effect of Income", fontsize=14)
axes[1].set_xticklabels(['Able to Pay', 'Financial Toxicity (Unable/Delay)'])
axes[1].set_ylabel("Family Income ($)")

plt.tight_layout()
plt.show()


# --- VISUALIZATION 2: The "Comorbidity Multiplier" ---
# Does having Diabetes or Heart Disease ALONG with Cancer increase financial risk?

# 2. Calculate the "Toxicity Rate" for each group
risk_data = []
baseline_rate = clean_df['UNABLE'].mean() * 100 # The average risk for ANY cancer patient

for code, name in disease_map.items():
    if code in clean_df.columns:
        # Get patients with this specific disease
        subset = clean_df[clean_df[code] == 1]
        # Calculate % who are UNABLE
        risk = subset['UNABLE'].mean() * 100
        risk_data.append({'Condition': name, 'Risk_Percentage': risk})

# 3. Plot
risk_df = pd.DataFrame(risk_data).sort_values('Risk_Percentage', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=risk_df, x='Risk_Percentage', y='Condition', palette="magma")
plt.axvline(x=baseline_rate, color='red', linestyle='--', label=f'Avg Cancer Patient ({baseline_rate:.1f}%)')
plt.title("Financial Toxicity Rate by Comorbidity", fontsize=14)
plt.xlabel("Percentage of Patients Reporting Financial Issues (%)")
plt.legend()
plt.show()