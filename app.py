
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="🎓",
    layout="wide"
)

# 2. Load Model and Scaler
def load_artifacts():
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Files not found! Make sure 'best_model.pkl' and 'scaler.pkl' are in the same folder.")
        return None, None

model, scaler = load_artifacts()

# 3. EXACT Feature Names (26 Columns in the specific order from your DataFrame)
feature_names = [
    'Marital status', 
    'Application mode', 
    'Application order', 
    'Course', 
    'Daytime/evening attendance', 
    'Previous qualification', 
    "Mother's qualification", 
    "Father's qualification", 
    "Mother's occupation", 
    'Displaced', 
    'Educational special needs', 
    'Debtor', 
    'Tuition fees up to date', 
    'Gender', 
    'Scholarship holder', 
    'Age at enrollment', 
    'Curricular units 1st sem (without evaluations)', 
    'Curricular units 2nd sem (credited)',      # Kept in your code
    'Curricular units 2nd sem (enrolled)',      # Kept in your code
    'Curricular units 2nd sem (evaluations)',   # Kept in your code
    'Curricular units 2nd sem (approved)',      # Kept in your code
    'Curricular units 2nd sem (grade)',         # Kept in your code
    'Curricular units 2nd sem (without evaluations)', # Kept in your code
    'Unemployment rate', 
    'Inflation rate', 
    'GDP'
]

# 4. Mappings (Only for Dropdowns)
marital_status_map = {
    1: "1 – Single", 2: "2 – Married", 3: "3 – Widower", 4: "4 – Divorced", 
    5: "5 – Facto union", 6: "6 – Legally separated"
}

parent_qual_map = {
    1: "1 - Secondary Education - 12th Year", 2: "2 - Higher Education - Bachelor's", 3: "3 - Higher Education - Degree",
    4: "4 - Higher Education - Master's", 5: "5 - Higher Education - Doctorate", 6: "6 - Frequency of Higher Education",
    9: "9 - 12th Year - Not Completed", 10: "10 - 11th Year - Not Completed", 11: "11 - 7th Year (Old)",
    12: "12 - Other - 11th Year", 13: "13 - 2nd year complementary HS", 14: "14 - 10th Year",
    18: "18 - General commerce", 19: "19 - Basic Education 3rd Cycle", 20: "20 - Complementary HS",
    22: "22 - Technical-professional", 25: "25 - Complementary HS - not concluded", 26: "26 - 7th year",
    27: "27 - 2nd cycle general HS", 29: "29 - 9th Year - Not Completed", 30: "30 - 8th year",
    31: "31 - General Course Admin", 33: "33 - Supp. Accounting/Admin", 34: "34 - Unknown",
    35: "35 - Can't read or write", 36: "36 - Can read without 4th year", 37: "37 - Basic education 1st cycle",
    38: "38 - Basic Education 2nd Cycle", 39: "39 - Technological specialization", 40: "40 - Higher education - degree",
    41: "41 - Specialized higher studies", 42: "42 - Professional higher technical", 43: "43 - Master (2nd cycle)",
    44: "44 - Doctorate (3rd cycle)"
}

mother_occ_map = {
    0: "0 - Student", 1: "1 - Legislative/Directors", 2: "2 - Intellectual/Scientific",
    3: "3 - Intermediate Technicians", 4: "4 - Administrative staff", 5: "5 - Personal Services/Security",
    6: "6 - Farmers/Skilled Ag", 7: "7 - Skilled Industry/Construction",
    8: "8 - Installation/Machine Ops", 9: "9 - Unskilled Workers", 10: "10 - Armed Forces", 90: "90 - Other",
    99: "99 - (blank)", 122: "122 - Health pro", 123: "123 - Teachers", 125: "125 - ICT Specialists",
    131: "131 - Intermediate Science/Eng", 132: "132 - Intermediate Health",
    134: "134 - Intermediate Legal/Social", 141: "141 - Office workers",
    143: "143 - Data/Financial ops", 144: "144 - Other admin", 151: "151 - Personal service",
    152: "152 - Sellers", 153: "153 - Personal care", 171: "171 - Skilled construction",
    173: "173 - Skilled printing", 175: "175 - Food/Wood/Clothing", 191: "191 - Cleaning",
    192: "192 - Unskilled Ag", 193: "193 - Unskilled Const", 194: "194 - Meal prep"
}

yes_no_map = {1: "1 – Yes", 0: "0 – No"}
gender_map = {1: "1 – Male", 0: "0 – Female"}
attendance_map = {1: "1 – Daytime", 0: "0 – Evening"}

def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key
    return 0

# 5. Application Layout
st.title("🎓 Student Success Predictor")
st.markdown("Enter student details below to predict the likelihood of Dropout or Graduation.")

if model and scaler:
    with st.form("prediction_form"):
        # --- SECTION 1: Personal & Demographic ---
        st.subheader("📝 Personal & Demographic")
        c1, c2, c3 = st.columns(3)
        with c1:
            marital_status = get_key(st.selectbox("Marital Status", list(marital_status_map.values())), marital_status_map)
            gender = get_key(st.selectbox("Gender", list(gender_map.values())), gender_map)
            age_at_enrollment = st.number_input("Age at Enrollment", min_value=17, max_value=80, value=20)
        with c2:
            displaced = get_key(st.selectbox("Displaced (Living away)", list(yes_no_map.values())), yes_no_map)
            educational_needs = get_key(st.selectbox("Educational Special Needs", list(yes_no_map.values())), yes_no_map)
        with c3:
            debtor = get_key(st.selectbox("Debtor", list(yes_no_map.values())), yes_no_map)
            tuition_fees = get_key(st.selectbox("Tuition Fees Up to Date", list(yes_no_map.values())), yes_no_map)
            scholarship = get_key(st.selectbox("Scholarship Holder", list(yes_no_map.values())), yes_no_map)

        st.divider()

        # --- SECTION 2: Application & Course (Manual Entry as requested) ---
        st.subheader("🏫 Application & Course")
        c4, c5 = st.columns(2)
        with c4:
            app_mode = st.number_input("Application Mode (Code)", min_value=0, step=1, help="e.g. 1, 17, 44")
            app_order = st.number_input("Application Order (0-9)", min_value=0, max_value=9, step=1)
            course = st.number_input("Course (Code)", min_value=0, step=1, help="e.g. 9119, 9003")
        with c5:
            attendance = get_key(st.selectbox("Attendance Time", list(attendance_map.values())), attendance_map)
            prev_qual = st.number_input("Previous Qualification (Code)", min_value=0, step=1, help="e.g. 1, 40")

        st.divider()

        # --- SECTION 3: Socio-Economic ---
        st.subheader("💼 Socio-Economic")
        c6, c7, c8 = st.columns(3)
        with c6:
            mother_qual = get_key(st.selectbox("Mother's Qualification", list(parent_qual_map.values())), parent_qual_map)
            mother_occ = get_key(st.selectbox("Mother's Occupation", list(mother_occ_map.values())), mother_occ_map)
        with c7:
            father_qual = get_key(st.selectbox("Father's Qualification", list(parent_qual_map.values())), parent_qual_map)
        with c8:
            unemployment = st.number_input("Unemployment Rate (%)", format="%.1f", value=10.0)
            inflation = st.number_input("Inflation Rate (%)", format="%.1f", value=1.0)
            gdp = st.number_input("GDP", format="%.1f", value=0.0)

        st.divider()

        # --- SECTION 4: Academic Performance (7 Columns Total) ---
        st.subheader("📊 Academic Performance")
        c9, c10 = st.columns(2)
        
        with c9:
            st.markdown("**1st Semester**")
            cu_1st_without = st.number_input("Units 1st Sem (Without Evals)", min_value=0, step=1)
            
        with c10:
            st.markdown("**2nd Semester**")
            cu_2nd_credited = st.number_input("Units 2nd Sem (Credited)", min_value=0, step=1)
            cu_2nd_enrolled = st.number_input("Units 2nd Sem (Enrolled)", min_value=0, step=1)
            cu_2nd_eval = st.number_input("Units 2nd Sem (Evaluations)", min_value=0, step=1)
            cu_2nd_approved = st.number_input("Units 2nd Sem (Approved)", min_value=0, step=1)
            cu_2nd_grade = st.number_input("Units 2nd Sem (Grade)", format="%.2f", value=12.0)
            cu_2nd_without = st.number_input("Units 2nd Sem (Without Evals)", min_value=0, step=1)

        submit_btn = st.form_submit_button("🚀 Predict Outcome", type="primary")

    if submit_btn:
        # Construct DataFrame with exactly 26 columns in the correct order
        input_list = [
            marital_status,      
            app_mode,            
            app_order,           
            course,              
            attendance,          
            prev_qual,           
            mother_qual,         
            father_qual,         
            mother_occ,          
            displaced,           
            educational_needs,   
            debtor,              
            tuition_fees,        
            gender,              
            scholarship,         
            age_at_enrollment,   
            cu_1st_without,      # 1st Sem
            cu_2nd_credited,     # 2nd Sem Start
            cu_2nd_enrolled,     
            cu_2nd_eval,
            cu_2nd_approved,   
            cu_2nd_grade,        
            cu_2nd_without,      # 2nd Sem End
            unemployment,        
            inflation,           
            gdp                  
        ]
        
        input_data = pd.DataFrame([input_list], columns=feature_names)

        try:
            # 1. Scale
            scaled_data = scaler.transform(input_data)
            
            # 2. Predict
            prediction = model.predict(scaled_data)[0]
            
            # 3. Display
            st.markdown("---")
            st.subheader("Prediction Result")
            
            # Note: In your training code, you mapped 'Graduate': 0, 'Dropout': 1
            if prediction == 1: 
                 st.error(f"Prediction: DROPOUT 🔴")
            else:
                 st.success(f"Prediction: GRADUATE 🟢")

        except Exception as e:
            st.error(f"Error: {e}")