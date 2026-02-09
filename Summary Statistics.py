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
demog_features = ["FAMINC", "TOTSLF", "AGELAST", "SEX"]
cancer_features = ["CABLADDR", "CABREAST", "CACERVIX", "CACOLON", "CALUNG", "CALYMPH", "CAMELANO", "CAOTHER"]
clean_df = main_df[main_df['CANCERDX'] == 1].copy() 
clean_df = clean_df[(clean_df[demog_features] >= 0).all(axis=1)]

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
# SUMMARY STATISTICS
# ==========================================
print("--- SUMMARY STATISTICS (For Proposal) ---")
print(clean_df[["TOTSLF", "FAMINC", "UNABLE"]].describe())

print("\n--- THE SUBSIDY SIGNAL: Quitting Rates by Group ---")
# This table proves if Public Insurance is working better than Uninsured
policy_stats = clean_df.groupby('POLICY_GROUP')[['UNABLE', 'TOTSLF', 'FAMINC']].mean()
print(policy_stats)

# ==========================================
# 2. NEW CHART: Cancer Type vs Quitting Count
# ==========================================
print("\n--- GENERATING BAR CHART: Cancer Types vs Quitting Treatment ---")

cancer_counts = []

for c_type in cancer_features:
    # Filter for patients who have this specific cancer type (Assuming 1 = Yes)
    patients_with_cancer = clean_df[clean_df[c_type] == 1]
    
    # Calculate how many are 'UNABLE' (Quitting)
    quit_count = patients_with_cancer['UNABLE'].sum()
    
    # Optional: Calculate percentage to see rate as well
    total_patients = len(patients_with_cancer)
    quit_rate = (quit_count / total_patients * 100) if total_patients > 0 else 0
    
    cancer_counts.append({
        'Cancer Type': c_type.replace('CA', ''), # Clean up name (e.g., CABREAST -> BREAST)
        'Quitting Count': quit_count,
        'Total Patients': total_patients,
        'Quitting Rate': quit_rate
    })

# Convert to DataFrame for plotting
viz_df = pd.DataFrame(cancer_counts)
viz_df = viz_df.sort_values(by='Quitting Count', ascending=False)

# Plotting
plt.figure(figsize=(12, 6))
sns.barplot(data=viz_df, x='Cancer Type', y='Quitting Count', palette='viridis')

plt.title('Number of People Quitting Treatment (Unable to Pay) by Cancer Type', fontsize=14)
plt.xlabel('Cancer Type', fontsize=12)
plt.ylabel('Number of People', fontsize=12)
plt.xticks(rotation=45)

# Add value labels on top of bars
for i, row in enumerate(viz_df.itertuples()):
    plt.text(i, row._2 + 0.1, f'{int(row._2)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

print("\nData used for Chart:")
print(viz_df[['Cancer Type', 'Quitting Count', 'Total Patients', 'Quitting Rate']])

