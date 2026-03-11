import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
import warnings

# Suppress sklearn warnings in the cloud environment
warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------------------------------------------------------
# PAGE SETUP & CLINICAL THEME
# -------------------------------------------------------------------------
st.set_page_config(page_title="Medicaid Allocation System", layout="wide", initial_sidebar_state="expanded")

# Inject Custom CSS for a Hospital Vibe
st.markdown("""
    <style>
    /* Medical Blue Headers */
    h1, h2, h3 {
        color: #004080;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        border-bottom: 1px solid #E0E0E0;
        padding-bottom: 4px;
        margin-bottom: 15px;
    }
    /* Action Buttons */
    .stButton>button {
        background-color: #004080;
        color: white;
        border-radius: 4px;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #00264d;
        color: white;
    }
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        color: #8B0000;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_data():
    return joblib.load('meps_model_data.pkl')

try:
    model_data = load_model_data()
    final_rf_model = model_data['model']
    selected_features_list = model_data['selected_features']
    X_train_selected = model_data['X_train_selected']
except FileNotFoundError:
    st.error("SYSTEM ERROR: 'meps_model_data.pkl' not found. Please verify repository files.")
    st.stop()

# -------------------------------------------------------------------------
# UI FRONTEND - CLINICAL DASHBOARD
# -------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📋 Chart Details")
    st.text_input("Patient ID (MRN)", value="PT-84729-A", disabled=True)
    st.text_input("Attending", value="Dr. G. House", disabled=True)
    st.date_input("Date of Assessment")
    st.divider()
    st.markdown("### System Status")
    st.success("Model: Active\nDB: MEPS-23 Linked")

st.title("Proactive Medicaid Resource Allocation System")
st.markdown("**Department of Oncology | Financial Toxicity & Adherence Prevention Unit**")
st.divider()

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("### 👤 I. Demographics & Coverage")
    patient_age = st.number_input("Age (Years)", min_value=0, max_value=120, value=55)
    patient_sex = st.radio("Assigned Sex at Birth", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
    
    region_choice = st.selectbox("Primary Residence (Region)", options=["Northeast", "Midwest", "South", "West"])
    region_map = {"Northeast": 1, "Midwest": 2, "South": 3, "West": 4}
    region_idx = region_map[region_choice]
    
    st.markdown("**Active Coverage Flags**")
    patient_vet = 1 if st.checkbox("Veterans Affairs (VA)") else 0
    patient_mil = 1 if st.checkbox("Tricare / Military") else 0
    patient_fed = 1 if st.checkbox("Federal Employee Health Benefits") else 0

with col2:
    st.markdown("### 📊 II. Socioeconomic Index")
    patient_faminc = st.number_input("Reported Family Income (Annual $)", min_value=0.0, value=45000.0, step=1000.0)
    patient_totslf = st.number_input("Current Out-of-Pocket Burden ($)", min_value=0.0, value=2500.0, step=100.0)
    
    patient_famsze = st.number_input("Household Size", min_value=1, value=2)
    patient_pov = st.slider("Federal Poverty Level (Proxy)", min_value=1, max_value=5, value=3, help="1=Poor, 5=High Income")
    
    st.markdown("**SDoH Risk Factors**")
    patient_foodst = 1 if st.checkbox("Food Insecurity (SNAP)") else 2
    patient_adl = 1 if st.checkbox("ADL Assistance Required") else 2
    patient_ddnwrk = st.number_input("Days Missed Work (Illness)", min_value=0, value=0)
    patient_phq2 = st.slider("PHQ-2 Depression Screener", min_value=0, max_value=6, value=0)

with col3:
    st.markdown("### 🩺 III. Clinical Presentation")
    cancer_list = ["Bladder", "Breast", "Cervix", "Colon", "Lung", "Lymphoma", "Melanoma", "Other", "Prostate", "Skin (Non-Melanoma)", "Skin (Unknown)", "Uterus", "None"]
    # Defaulting to Colon (index 3) for quick testing
    cancer_choice = st.selectbox("Primary Oncology Diagnosis", options=cancer_list, index=3) 
    
    disease_list = ["Diabetes", "High Blood Pressure", "Coronary Heart Disease", "Angina", "Heart Attack", "Other Heart Disease", "Stroke", "High Cholesterol", "Emphysema", "Asthma", "Chronic Bronchitis", "Arthritis"]
    selected_diseases = st.multiselect("Documented Comorbidities", options=disease_list)
    
    st.markdown("**12-Month Utilization History**")
    patient_ipdis = st.number_input("Inpatient Admissions", min_value=0, value=0)
    patient_ipngtd = st.number_input("Total Inpatient Days", min_value=0, value=0)
    patient_ertot = st.number_input("Emergency Department Visits", min_value=0, value=0)

# -------------------------------------------------------------------------
# BACKEND CALCULATION LOGIC
# -------------------------------------------------------------------------
st.divider()
if st.button("RUN STATISTICAL SUBSIDY EXPECTATION", type="primary", use_container_width=True):
    with st.spinner("Querying model and calculating risk profile..."):
        
        # 1. Base Variables
        catastrophic_cost = 1 if patient_totslf > (0.10 * patient_faminc) else 0
        patient_medicare = 1 if patient_age >= 65 else 0
        patient_chip = 1 if patient_age <= 19 else 0
        
        # 2. Cancer Mapping
        cancer_features = ["CABLADDR", "CABREAST", "CACERVIX", "CACOLON", "CALUNG", "CALYMPH", "CAMELANO", "CAOTHER", "CAPROSTA", "CASKINNM", "CASKINDK", "CAUTERUS"]
        patient_cancers = {col: 2 for col in cancer_features}
        if cancer_choice != "None":
            selected_idx = cancer_list.index(cancer_choice)
            patient_cancers[cancer_features[selected_idx]] = 1

        # 3. Comorbidity Mapping
        disease_features = ["DIABDX", "HIBPDX", "CHDDX", "ANGIDX", "MIDX", "OHRTDX", "STRKDX", "CHOLDX", "EMPHDX", "ASTHDX", "CHBRON", "ARTHDX"]
        patient_diseases = {col: 2 for col in disease_features}
        
        # FIX 1: Generate the exact AGE_DIAG names the model expects
        patient_age_diag = {}
        for col in disease_features:
            if col != "CHBRON": # CHBRON doesn't have an age variable in MEPS
                patient_age_diag[f"AGE_DIAG_{col}"] = 0
                
        for d in selected_diseases:
            idx = disease_list.index(d)
            feat_name = disease_features[idx]
            patient_diseases[feat_name] = 1
            if feat_name != "CHBRON":
                patient_age_diag[f"AGE_DIAG_{feat_name}"] = patient_age 
            
        # 4. Engineered Utilizations
        total_visits_calc = patient_ertot 
        inpatient_burden_calc = patient_ipdis + patient_ipngtd
        care_intensity_calc = total_visits_calc + inpatient_burden_calc
        er_dependency_calc = patient_ertot / (total_visits_calc + 1e-6)
        
        has_cancer = 1 if cancer_choice != "None" else 0
        cancer_dep_calc = 1 if (has_cancer == 1 and patient_phq2 > 2) else 0
        disease_count = len(selected_diseases)
        elderly_multi_calc = 1 if (patient_age >= 65 and disease_count >= 2) else 0
        fin_spiral_calc = catastrophic_cost * patient_ddnwrk
        
        pov_reverse = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}.get(patient_pov, 3)
        sdoh_score_calc = pov_reverse + (2 if patient_foodst == 1 else 0) + (2 if patient_adl == 1 else 0)
        avg_nights_calc = patient_ipngtd / max(1, patient_ipdis)

        # 5. Build the Master Dictionary
        patient_data = {
            'FAMINC': [patient_faminc], 'TOTSLF': [patient_totslf], 'CATASTROPHIC_COST': [catastrophic_cost],
            'AGELAST': [patient_age], 'SEX': [patient_sex], 'IS_MEDICARE_AGE': [patient_medicare],
            'IS_CHIP_AGE': [patient_chip], 'IS_VETERAN': [patient_vet], 'IS_MILITARY_FAM': [patient_mil],
            'IS_FED_WORKER': [patient_fed], 'FAMILY_SIZE': [patient_famsze],
            'INCOME_PER_CAPITA': [patient_faminc / max(1, patient_famsze)],
            'POVERTY_CATEGORY': [patient_pov], 'FOOD_STAMPS': [patient_foodst],
            'DAYS_MISSED_WORK': [patient_ddnwrk], 'ADL_HELP_NEEDED': [patient_adl],
            'PHQ2_DEPRESSION_SCORE': [patient_phq2],
            'REGION_NORTHEAST': [1 if region_idx == 1 else 0],
            'REGION_MIDWEST': [1 if region_idx == 2 else 0],
            'REGION_SOUTH': [1 if region_idx == 3 else 0],
            'REGION_WEST': [1 if region_idx == 4 else 0],
            'TOTAL_VISITS': [total_visits_calc], 'INPATIENT_BURDEN': [inpatient_burden_calc],
            'CARE_INTENSITY_INDEX': [care_intensity_calc], 'ER_DEPENDENCY': [er_dependency_calc],
            'CANCER_AND_DEPRESSION': [cancer_dep_calc], 'ELDERLY_MULTIMORBIDITY': [elderly_multi_calc],
            'FINANCIAL_SPIRAL_RISK': [fin_spiral_calc], 'SDOH_VULNERABILITY_SCORE': [sdoh_score_calc],
            'AVG_NIGHTS_PER_STAY': [avg_nights_calc]
        }
        
        patient_data.update({k: [v] for k, v in patient_cancers.items()})
        patient_data.update({k: [v] for k, v in patient_diseases.items()})
        patient_data.update({k: [v] for k, v in patient_age_diag.items()})
        
        # FIX 2: Override any corrupt pickle lists with the model's actual memory
        if hasattr(final_rf_model, "feature_names_in_"):
            expected_features = final_rf_model.feature_names_in_
        else:
            expected_features = selected_features_list
            
        # Fill any missing baseline features with generic safeties 
        for col in expected_features:
            if col not in patient_data:
                patient_data[col] = [0]
                
        # 6. Predict using exact matched features
        new_patient_df = pd.DataFrame(patient_data)[expected_features]
        new_patient_df_selected = new_patient_df.values
        
        # Medicaid Gate
        approx_fpl = 14580 + (5140 * (max(1, patient_famsze) - 1))
        medicaid_income_limit = approx_fpl * 1.38 
        is_medicaid_eligible = True
        
        if patient_age > 19 and patient_faminc > medicaid_income_limit:
            if patient_adl == 2 and patient_medicare == 0:
                is_medicaid_eligible = False
        if patient_chip == 1 and patient_faminc > (approx_fpl * 3.0):
            is_medicaid_eligible = False

        st.markdown("### 📋 Clinical Summary & Results")
        if is_medicaid_eligible:
            recommended_subsidy = final_rf_model.predict(new_patient_df_selected)[0]
            recommended_subsidy = max(0, recommended_subsidy)
            st.success(f"### Computed Statistical Subsidy Expectation: ${recommended_subsidy:,.2f}")
            st.caption("Estimated public burden based on full-year patient realities and historical MEPS data.")
        else:
            st.error("### Computed Statistical Subsidy Expectation: $0.00")
            st.caption("FLAG: Patient income exceeds Medicaid/CHIP eligibility thresholds for their demographic.")

        # 7. SHAP Plot
        st.markdown("#### Cost Drivers Analysis (SHAP Breakdown)")
        explainer = shap.Explainer(final_rf_model, X_train_selected)
        shap_values = explainer(new_patient_df)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], show=False)
        plt.tight_layout()
        st.pyplot(fig)