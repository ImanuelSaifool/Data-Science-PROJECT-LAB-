import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier #for categorical data, you wanna use RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, accuracy_score
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------------------------------
# dataset's entrance into this world
df2021p1 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2021_data_part1.csv")
df2021p2 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2021_data_part2.csv")
df2022 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2022%20data.csv")
df2023 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2023%20data.csv")

# renaming stuff
df2021p1 = df2021p1.rename(columns={"TOTSLF21": "TOTSLF"})
df2021p2 = df2021p2.rename(columns={"TOTSLF21": "TOTSLF"}) 
df2022 = df2022.rename(columns={"TOTSLF22": "TOTSLF"})
df2023 = df2023.rename(columns={"TOTSLF23": "TOTSLF"})

# combining them
main_df = pd.concat([df2021p1, df2021p2, df2022, df2023], axis=0)

# data filter
clean_df = main_df[main_df['CANCERDX'] == 1].copy() 

def is_unable(row):
    # Check all three rounds
    rounds = [row.get('UNABLE31'), row.get('UNABLE42'), row.get('UNABLE53')]
    
    # If ANY round is 1 (Yes), they faced a barrier
    if 1 in rounds:
        return 1
    elif all(r == 2 or pd.isna(r) or r < 0 for r in rounds): 
        return 0
    else:
        return np.nan
# ----------------------------------------------------------------------------------------------------------------------------------------------
# ==========================================
# FIRST: Finding the relationship between age and quitting
# ==========================================
clean_df['HIT_BARRIER'] = clean_df.apply(is_unable, axis=1)

# drop rows where we don't know the answer
clean_df = clean_df.dropna(subset=['HIT_BARRIER', 'AGELAST'])

# defining variables and doing a logistic regression
X1 = clean_df[["AGELAST"]]
y1 = clean_df["HIT_BARRIER"]

log_reg = LogisticRegression()
log_reg.fit(X1, y1)

# presenting the result
print(f"Age Coefficient: {log_reg.coef_[0][0]:.4f}")

sns.regplot(x='AGELAST', y='HIT_BARRIER', data=clean_df, 
            logistic=True, 
            ci=95, # This adds a shaded 'confidence interval' (very good for your report!)
            scatter_kws={'alpha':0.05, 'color': 'navy'}, # Makes the raw data points transparent
            line_kws={'color': 'red', 'lw': 3}) # Makes the regression line bold and red

# Labeling of the graph
plt.title('Probability of Facing Treatment Barriers vs. Age', fontsize=14)
plt.xlabel('Age of Patient', fontsize=12)
plt.ylabel('Probability of Hitting a Barrier (0.0 to 1.0)', fontsize=12)
plt.ylim(-0.05, 1.05) # Keeps the y-axis strictly between 0 and 1
plt.show()

# ----------------------------------------------------------------------------------------------------------------------------------------------
# =========================================================
# SECOND: Finding the relationship between sex and quitting
# =========================================================
clean_df['HIT_BARRIER'] = clean_df.apply(is_unable, axis=1)

# drop rows where we don't know the answer
clean_df = clean_df.dropna(subset=['HIT_BARRIER', 'SEX'])

# defining variables and doing a logistic regression
X2 = clean_df[["SEX"]]
y2 = clean_df["HIT_BARRIER"]

log_reg = LogisticRegression()
log_reg.fit(X2, y2)

# presenting the result
print(f"Sex Coefficient: {log_reg.coef_[0][0]:.4f}")

sns.regplot(x='SEX', y='HIT_BARRIER', data=clean_df, 
            logistic=True, 
            ci=95, # This adds a shaded 'confidence interval' (very good for your report!)
            scatter_kws={'alpha':0.05, 'color': 'navy'}, # Makes the raw data points transparent
            line_kws={'color': 'red', 'lw': 3}) # Makes the regression line bold and red

# Labeling of the graph
plt.title('Probability of Facing Treatment Barriers vs. Sex', fontsize=14)
plt.xlabel('Sex of Patient', fontsize=12)
plt.ylabel('Probability of Hitting a Barrier (0.0 to 1.0)', fontsize=12)
plt.ylim(-0.05, 1.05) # Keeps the y-axis strictly between 0 and 1
plt.show()
# ----------------------------------------------------------------------------------------------------------------------------------------------
# ===================================================================
# THIRD: Finding the relationship between family income and quitting
# ===================================================================

clean_df['HIT_BARRIER'] = clean_df.apply(is_unable, axis=1)

# drop rows where we don't know the answer
clean_df = clean_df.dropna(subset=['HIT_BARRIER', 'FAMINC'])

# defining variables and doing a logistic regression
X3 = clean_df[["FAMINC"]]
y3= clean_df["HIT_BARRIER"]

log_reg = LogisticRegression()
log_reg.fit(X3, y3)

# presenting the result
print(f"Family Income Coefficient: {log_reg.coef_[0][0]:.4f}")

sns.regplot(x='FAMINC', y='HIT_BARRIER', data=clean_df, 
            logistic=True, 
            ci=95, # This adds a shaded 'confidence interval' (very good for your report!)
            scatter_kws={'alpha':0.05, 'color': 'navy'}, # Makes the raw data points transparent
            line_kws={'color': 'red', 'lw': 3}) # Makes the regression line bold and red

# Labeling of the graph
plt.title('Probability of Facing Treatment Barriers vs. Family Income', fontsize=14)
plt.xlabel('Family Income of Patient', fontsize=12)
plt.ylabel('Probability of Hitting a Barrier (0.0 to 1.0)', fontsize=12)
plt.ylim(-0.05, 1.05) # Keeps the y-axis strictly between 0 and 1
plt.show()

# ----------------------------------------------------------------------------------------------------------------------------------------------

# ==========================================
# FOURTH: Finding the relationship between out of pocket cost and quitting
# ==========================================

clean_df['HIT_BARRIER'] = clean_df.apply(is_unable, axis=1)

# drop rows where we don't know the answer
clean_df = clean_df.dropna(subset=['HIT_BARRIER', 'TOTSLF'])

# defining variables and doing a logistic regression
X4 = clean_df[["TOTSLF"]]
y4= clean_df["HIT_BARRIER"]

log_reg = LogisticRegression()
log_reg.fit(X4, y4)

# presenting the result
print(f"Out of Pocket Cost Coefficient: {log_reg.coef_[0][0]:.4f}")

sns.regplot(x='TOTSLF', y='HIT_BARRIER', data=clean_df, 
            logistic=True, 
            ci=95, # This adds a shaded 'confidence interval' (very good for your report!)
            scatter_kws={'alpha':0.05, 'color': 'navy'}, # Makes the raw data points transparent
            line_kws={'color': 'red', 'lw': 3}) # Makes the regression line bold and red

# Labeling of the graph
plt.title('Probability of Facing Treatment Barriers vs. Out of Pocket Cost', fontsize=14)
plt.xlabel('Out of Pocket Cost', fontsize=12)
plt.ylabel('Probability of Hitting a Barrier (0.0 to 1.0)', fontsize=12)
plt.ylim(-0.05, 1.05) # Keeps the y-axis strictly between 0 and 1
plt.show()