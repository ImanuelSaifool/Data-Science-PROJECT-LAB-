import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------------------------------
# Data's entrance into this teeny weeny world
# ----------------------------------------------------------------------------------------------------------------------------------------------
df2021p1 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2021_data_part1.csv")
df2021p2 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2021_data_part2.csv")
df2022 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2022%20data.csv")
df2023 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2023%20data.csv")

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 2. Standardizing column names
    # we do this so that we can easily integrate multiple data files without changing the name on the raw data
# ----------------------------------------------------------------------------------------------------------------------------------------------
# Out of pocket cost
df2021p1 = df2021p1.rename(columns={"TOTSLF21": "TOTSLF"})
df2021p2 = df2021p2.rename(columns={"TOTSLF21": "TOTSLF"})
df2022 = df2022.rename(columns={"TOTSLF22": "TOTSLF"})
df2023 = df2023.rename(columns={"TOTSLF23": "TOTSLF"})

# Family income
df2021p1 = df2021p1.rename(columns={"FAMINC21": "FAMINC"})
df2021p2 = df2021p2.rename(columns={"FAMINC21": "FAMINC"})
df2022 = df2022.rename(columns={"FAMINC22": "FAMINC"})
df2023 = df2023.rename(columns={"FAMINC23": "FAMINC"})

# Insurance covered
df2021p1 = df2021p1.rename(columns={"INSCOV21": "INSCOV"})
df2021p2 = df2021p2.rename(columns={"INSCOV21": "INSCOV"})
df2022 = df2022.rename(columns={"INSCOV22": "INSCOV"})
df2023 = df2023.rename(columns={"INSCOV23": "INSCOV"})

# Renaming Medicare
df2021p1 = df2021p1.rename(columns={"TOTMCR21": "TOTMCR"})
df2021p2 = df2021p2.rename(columns={"TOTMCR21": "TOTMCR"})
df2022 = df2022.rename(columns={"TOTMCR22": "TOTMCR"})
df2023 = df2023.rename(columns={"TOTMCR23": "TOTMCR"})

# Renaming Medicaid
df2021p1 = df2021p1.rename(columns={"TOTMCD21": "TOTMCD"})
df2021p2 = df2021p2.rename(columns={"TOTMCD21": "TOTMCD"})
df2022 = df2022.rename(columns={"TOTMCD22": "TOTMCD"})
df2023 = df2023.rename(columns={"TOTMCD23": "TOTMCD"})

# Renaming Veterans Affair
df2021p1 = df2021p1.rename(columns={"TOTVA21": "TOTVA"})
df2021p2 = df2021p2.rename(columns={"TOTVA21": "TOTVA"})
df2022 = df2022.rename(columns={"TOTVA22": "TOTVA"})
df2023 = df2023.rename(columns={"TOTVA23": "TOTVA"})

#Renaming Other Federal
df2021p1 = df2021p1.rename(columns={"TOTOFD21": "TOTOFD"})
df2021p2 = df2021p2.rename(columns={"TOTOFD21": "TOTOFD"})
df2022 = df2022.rename(columns={"TOTOFD22": "TOTOFD"})
df2023 = df2023.rename(columns={"TOTOFD23": "TOTOFD"})

#Renaming State
df2021p1 = df2021p1.rename(columns={"TOTSTL21": "TOTSTL"})
df2021p2 = df2021p2.rename(columns={"TOTSTL21": "TOTSTL"})
df2022 = df2022.rename(columns={"TOTSTL22": "TOTSTL"})
df2023 = df2023.rename(columns={"TOTSTL23": "TOTSTL"})

#Renaming Worker's Comp
df2021p1 = df2021p1.rename(columns={"TOTWCP21": "TOTWCP"})
df2021p2 = df2021p2.rename(columns={"TOTWCP21": "TOTWCP"})
df2022 = df2022.rename(columns={"TOTWCP22": "TOTWCP"})
df2023 = df2023.rename(columns={"TOTWCP23": "TOTWCP"})


# Combining datasets
main_df = pd.concat([df2021p1, df2021p2, df2022, df2023], axis=0)

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 3. FILTERING & CLEANING
# ----------------------------------------------------------------------------------------------------------------------------------------------
# Define feature lists
demog_features = ["FAMINC", "TOTSLF", "AGELAST", "SEX"]
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

# Filter negative values for demographics only to prevent logic error
clean_df = clean_df[(clean_df[demog_features] >= 0).all(axis=1)]

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

# Create and Print Table
summary_table = pd.DataFrame(summary_list).sort_values(by="Prevalence (%)", ascending=False)
print("\n--- Comorbidity Impact Table ---")
print(summary_table.to_string(index=False))


# -------------------------------------------------------
# GRAPH A: Prevalence (How common is it?)
# -------------------------------------------------------
# 1. SETUP THE FIGURE (3 Subplots)
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
sns.set_style("whitegrid")

sns.barplot(
    data=summary_table, 
    x="Prevalence (%)", 
    y="Comorbidity", 
    ax=axes[0], 
    palette="Blues_r"
)
axes[0].set_title("Prevalence of Comorbidities\n(Frequency in Cancer Patients)", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Percentage of Patients (%)")
axes[0].set_ylabel("") # Remove y-label for cleaner look

# -------------------------------------------------------
# GRAPH B: Financial Toxicity (The "Problem")
# -------------------------------------------------------
# We sort this graph by Cost so the most expensive ones are at the top
cost_sorted = summary_table.sort_values(by="Avg OOP Cost ($)", ascending=False)

sns.barplot(
    data=cost_sorted, 
    x="Avg OOP Cost ($)", 
    y="Comorbidity", 
    ax=axes[1], 
    palette="Reds_r"
)
axes[1].set_title("Financial Toxicity\n(Avg. Out-of-Pocket Cost)", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Annual Cost ($)")
axes[1].set_ylabel("") 

# -------------------------------------------------------
# GRAPH C: Public Support (The "Policy Response")
# -------------------------------------------------------
# We sort this by Public Pay to see where the money is going
public_sorted = summary_table.sort_values(by="Avg Public Pay ($)", ascending=False)

sns.barplot(
    data=public_sorted, 
    x="Avg Public Pay ($)", 
    y="Comorbidity", 
    ax=axes[2], 
    palette="Greens_r"
)
axes[2].set_title("Government Support\n(Avg. Public Insurance Payment)", fontsize=14, fontweight='bold')
axes[2].set_xlabel("Annual Payment ($)")
axes[2].set_ylabel("")

# Final Layout Adjustments
plt.tight_layout()
plt.show()


# 1. SETUP (Focus on the Vulnerable Group)
# We look at people who might actually NEED the subsidy (Public + Uninsured/Low Income)
# (Assuming clean_df is already filtered for Cancer)

# 2. CREATE "FINANCIAL PRESSURE" VARIABLES
# This is what you want to "treat" with your subsidy
clean_df['OOP_TO_INCOME_RATIO'] = clean_df['TOTSLF'] / clean_df['FAMINC'].replace(0, 1) # Avoid div/0

# 3. SELECT FEATURES FOR HEATMAP
# We want to see if "Pressure" correlates with "Unadherence"
heatmap_cols = [
    'UNABLE',              # Target: 1 = Quit/Delayed, 0 = Stayed
    'TOTSLF',              # The raw cost
    'FAMINC',              # The raw ability to pay
    'OOP_TO_INCOME_RATIO', # The "Pressure" (The Gap)
    'PUBLIC_TOTAL'         # Current aid (Is it helping?)
]

# 4. PLOT
corr_matrix = clean_df[heatmap_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="RdBu_r", center=0)
plt.title("Diagnostics: Does Financial Pressure Cause Quitting?", fontsize=14)
plt.show()