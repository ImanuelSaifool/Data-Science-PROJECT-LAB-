import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier #for categorical data, you wanna use RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, accuracy_score
# ----------------------------------------------------------------------------------------------------------------------------------------------
# datasets
df2021p1 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2021_data_part1.csv")
df2021p2 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2021_data_part2.csv")
df2022 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2022_data.csv")
df2023 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2023_data.csv")

main_df = pd.concat([df2021p1, df2021p2, df2022, df2023], axis=0)

# data filter
cancer_df = main_df[main_df['CANCERDX'] == 1].copy() 
cancer_df = cancer_df[(cancer_df["FAMINC"] > 0) & (cancer_df["K6SUM42"] >= 0)]
# ----------------------------------------------------------------------------------------------------------------------------------------------
# =====================================================
# First: Evaluation of Causes for quitting
# =====================================================
# define variables
y1 = cancer_df["TOTSLF"] # for good luck, you define the y first
x1 = cancer_df[["CANCERDX"]]

# splitting training and testing data (pareto's rule)
X1_train, X1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.2, random_state=88)

# do the random forest gahh
rf1 = RandomForestRegressor(n_estimators=100, random_state=88)

# fit the data first
rf1.fit(X1_train, y1_train)
importances = rf1.feature_importances_
print(f"Feature Importances (Income vs Cost): {importances}")

avg_cost_cancer = main_df[main_df['CANCERDX']==1]['TOTSLF'].mean()
avg_cost_healthy = main_df[main_df['CANCERDX']==0]['TOTSLF'].mean()

print(f"Average Cost for Cancer Patients: ${avg_cost_cancer:,.2f}")
print(f"Average Cost for Others:          ${avg_cost_healthy:,.2f}")
print(f"Link Confirmed: Cancer patients pay {avg_cost_cancer/avg_cost_healthy:.1f}x more.")
#---------------------------------------------------------------------------------------------------------------------------------------------
# ==========================================
# FINAL VISUALIZATION
# ==========================================
# Let's plot the "Chain of Events"
links = ['Cancer -> $$', '$$ -> Depression', 'Depression -> Quit']
# For visualization, we use normalized scores or simple boolean confirmations
strengths = [100, rf2.feature_importances_[0]*100, acc_link3*100] 

plt.figure(figsize=(10, 5))
plt.plot(links, strengths, marker='o', linestyle='-', color='b')
plt.title("The Causal Chain: Path Analysis")