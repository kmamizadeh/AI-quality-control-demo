import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from typing import Dict, Any

# Paths to the files. These should be relative to the script location.
EXCEL_FILE = 'Polymer_Properties_Processed_by_python1.xlsx'
IMPACT_MODEL_FILE = 'regression_model.pkl'
TENSILE_MODEL_FILE = 'tensile_model.pkl'

# --- Custom CSS for a modern, clean look ---
st.markdown("""
<style>
    /* General body and typography */
    @font-face {
        font-family: 'Vazirmatn';
        font-style: normal;
        font-weight: 400;
        src: url('https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.0.3/fonts/web/Vazirmatn-Regular.woff2') format('woff2');
    }
    body {
        font-family: 'Vazirmatn', sans-serif;
        background-color: #f4f7f9;
        color: #333;
    }
    
    .stApp {
        background-color: #f4f7f9;
        color: #333; /* Ensuring all app text is dark */
    }

    /* Main title and headers */
    .centered-title h1, .centered-description p {
        text-align: center;
    }
    
    h1, h2, h3 {
        color: #2c3e50;
        text-align: right;
    }

    /* Explicitly setting a dark color for all text components */
    .stMarkdown, .stSelectbox, .stNumberInput, .stTextInput, .stCheckbox, .stButton {
        color: #333;
    }

    /* Input fields and selectboxes */
    .stTextInput input, .stNumberInput input {
        color: #333 !important; /* Force text color inside inputs */
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 10px;
        transition: all 0.2s ease-in-out;
    }
    
    /* Fix for Selectbox nesting issue: targets only the main visible container */
    .stSelectbox > div:first-child > div {
        color: #333 !important;
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 10px;
        transition: all 0.2s ease-in-out;
    }
    
    .stTextInput>div>div>input:focus, .stSelectbox>div>div:focus, .stNumberInput>div>div>input:focus {
        border-color: #007bff;
        box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25);
    }

    /* Buttons */
    .st-emotion-cache-192l57a { /* Main button styling */
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        transition: transform 0.2s ease-in-out;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    .st-emotion-cache-192l57a:hover {
        background-color: #0056b3;
        transform: translateY(-2px);
    }
    
    /* Success/Error/Warning messages */
    .stSuccess {
        background-color: #e6f7ee;
        border-left: 5px solid #28a745;
        color: #155724;
        border-radius: 8px;
        padding: 15px;
        margin-top: 10px;
    }
    .stError {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        color: #721c24;
        border-radius: 8px;
        padding: 15px;
        margin-top: 10px;
    }
    .stWarning {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        color: #856404;
        border-radius: 8px;
        padding: 15px;
        margin-top: 10px;
    }

    /* Spacing */
    .st-emotion-cache-1c881c1 {
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# --- Data and Model Loading (with caching) ---
@st.cache_data
def load_data_and_get_unique_values():
    """Loads the Excel data and extracts unique values for dropdowns."""
    if not os.path.exists(EXCEL_FILE):
        return None, None
    
    try:
        df = pd.read_excel(EXCEL_FILE)
        unique_values = {
            'Polymer1_Type': sorted(df['Polymer1_Type'].unique()),
            'Polymer2_Type': sorted(df['Polymer2_Type'].unique()),
            'Polymer3_Type': sorted(df['Polymer3_Type'].unique()),
            'Filler1_Type': sorted(df['Filler1_Type'].unique()),
            'Filler2_Type': sorted(df['Filler2_Type'].unique()),
            'Additive_Type': sorted(df['Additive_Type'].unique())
        }
        return df, unique_values
    except Exception as e:
        st.error(f"خطا در خواندن فایل اکسل: {e}")
        return None, None

@st.cache_resource
def load_model_and_get_columns():
    """Loads the trained models and their expected features."""
    try:
        if not os.path.exists(IMPACT_MODEL_FILE) or not os.path.exists(TENSILE_MODEL_FILE):
            st.error("فایل های مدل (.pkl) پیدا نشدند. لطفا مطمئن شوید که در مسیر صحیح قرار دارند.")
            return None, None, None, None
        
        impact_model = joblib.load(IMPACT_MODEL_FILE)
        tensile_model = joblib.load(TENSILE_MODEL_FILE)
        
        # Ensure models have feature_names_in_ attribute
        if not hasattr(impact_model, 'feature_names_in_') or not hasattr(tensile_model, 'feature_names_in_'):
            st.error("مدل‌ها به صورت صحیح آموزش داده نشده‌اند یا فاقد لیست ویژگی‌ها هستند.")
            return None, None, None, None

        impact_model_columns = impact_model.feature_names_in_.tolist()
        tensile_model_columns = tensile_model.feature_names_in_.tolist()

        return impact_model, tensile_model, impact_model_columns, tensile_model_columns
        
    except FileNotFoundError:
        st.error("خطا: فایل‌های مدل پیدا نشدند.")
        return None, None, None, None
    except Exception as e:
        st.error(f"خطا در بارگذاری مدل: {e}")
        return None, None, None, None

# --- Main Functions ---
def convert_impact_to_base(value, unit):
    """Converts impact values to the base unit of J/m."""
    if unit == "J/m":
        return value
    elif unit == "KJ/m":
        return value * 1000
    elif unit == "J/m^2":
        # This conversion is an approximation, as area is unknown.
        # Assuming a standard thickness or a value proportional to it.
        # For a simple example, we'll return the value as is.
        return value
    elif unit == "KJ/m^2":
        return value * 1000
    elif unit == "J/cm^2":
        return value * 10000
    return value

def convert_tensile_to_base(value, unit):
    """Converts tensile strength values to the base unit of MPa."""
    if unit == "MPa":
        return value
    elif unit == "GPa":
        return value * 1000
    elif unit == "Pa":
        return value / 1_000_000
    return value

def prepare_data_for_prediction(input_data: Dict[str, Any], model_columns: list) -> pd.DataFrame:
    """
    Prepares a DataFrame for prediction, ensuring all columns match the model's
    expected features, including one-hot encoded categorical variables.
    """
    # Create an empty DataFrame with all required columns, initialized to 0
    df_pred = pd.DataFrame(0, index=[0], columns=model_columns)

    for key, value in input_data.items():
        # Handle numerical columns directly
        if isinstance(value, (int, float)):
            if key in df_pred.columns:
                df_pred.loc[0, key] = value
            elif key + "_No break" in df_pred.columns and value == 1:
                # This handles the 'Impact_Not_Break' checkbox
                df_pred.loc[0, key + "_No break"] = 1

        # Handle categorical columns
        elif isinstance(value, str) and value: # check for non-empty string
            ohe_column_name = f"{key}_{value}"
            if ohe_column_name in df_pred.columns:
                df_pred.loc[0, ohe_column_name] = 1
            # Special case for Impact Test Type where we want to predict a specific value
            # It is likely already handled by the logic above.
            # It's good practice to keep the original column in a separate dictionary for clarity.

    return df_pred


# --- Prediction Logic ---
def predict_properties(data_to_predict: Dict[str, Any], impact_model, tensile_model, impact_cols: list, tensile_cols: list):
    """Makes predictions using the loaded models."""
    try:
        # Prepare data for Impact model prediction
        df_impact = prepare_data_for_prediction(data_to_predict, impact_cols)
        impact_pred = impact_model.predict(df_impact)[0]
        
        # Prepare data for Tensile model prediction
        df_tensile = prepare_data_for_prediction(data_to_predict, tensile_cols)
        tensile_pred = tensile_model.predict(df_tensile)[0]
        
        return {'impact': impact_pred, 'tensile': tensile_pred}
    except Exception as e:
        st.error(f"خطا در پیش‌بینی: {e}")
        return None


# --- Main App Structure ---
st.set_page_config(layout="wide", page_title="پیش‌بینی خواص پلیمر")
st.markdown("<div class='centered-title'><h1>🧪 برنامه پیش‌بینی و ثبت خواص کامپوزیت‌های پلیمری</h1></div>", unsafe_allow_html=True)
st.markdown(
    """
    <div class='centered-description'>
    <p>این برنامه به شما امکان می‌دهد خواص فرمولاسیون‌های پلیمری را ثبت کنید
    و بر اساس مدل‌های هوش مصنوعی، خواص نهایی آن‌ها را پیش‌بینی نمایید.</p>
    </div>
    """, unsafe_allow_html=True)

df, unique_values = load_data_and_get_unique_values()
impact_model, tensile_model, impact_cols, tensile_cols = load_model_and_get_columns()

col_form, col_predict = st.columns([1.5, 1])

with col_form:
    st.header("📝 ثبت اطلاعات جدید در دیتاست")

    with st.container():
        st.markdown("### ۱. مشخصات فرمولاسیون")
        
        st.markdown("**پلیمرها**")
        p1_type = st.selectbox("نوع پلیمر اول", options=[''] + (unique_values['Polymer1_Type'] if unique_values else []), key="p1_type_form")
        p1_perc = st.number_input("درصد پلیمر اول (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="p1_perc_form")
        p2_type = st.selectbox("نوع پلیمر دوم", options=[''] + (unique_values['Polymer2_Type'] if unique_values else []), key="p2_type_form")
        p2_perc = st.number_input("درصد پلیمر دوم (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="p2_perc_form")
        p3_type = st.selectbox("نوع پلیمر سوم", options=[''] + (unique_values['Polymer3_Type'] if unique_values else []), key="p3_type_form")
        p3_perc = st.number_input("درصد پلیمر سوم (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="p3_perc_form")
        
        st.markdown("---")

        st.markdown("**فیلرها**")
        f1_type = st.selectbox("نوع فیلر اول", options=[''] + (unique_values['Filler1_Type'] if unique_values else []), key="f1_type_form")
        f1_size = st.number_input("اندازه ذرات فیلر اول (میکرون)", min_value=0.0, key="f1_size_form")
        f1_perc = st.number_input("درصد فیلر اول (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="f1_perc_form")
        f2_type = st.selectbox("نوع فیلر دوم", options=[''] + (unique_values['Filler2_Type'] if unique_values else []), key="f2_type_form")
        f2_size = st.number_input("اندازه ذرات فیلر دوم (میکرون)", min_value=0.0, key="f2_size_form")
        f2_perc = st.number_input("درصد فیلر دوم (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="f2_perc_form")

        st.markdown("---")

        st.markdown("**افزودنی‌ها**")
        a_type = st.selectbox("نوع افزودنی", options=[''] + (unique_values['Additive_Type'] if unique_values else []), key="a_type_form")
        a_perc = st.number_input("درصد افزودنی (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="a_perc_form")
        a_func = st.selectbox("عملکرد افزودنی", options=[''] + ['Toughener', 'Impact Modifier', 'Colorant', 'Antioxidant', 'Unknown'], key="a_func_form")
        
        st.markdown("---")
        
        st.markdown("### ۲. نوع آزمون")
        impact_test_type = st.selectbox("نوع آزمون ضربه", options=[''] + ['Charpy', 'Izod'], key="impact_test_type_entry_form")
        impact_not_break = st.checkbox("شکسته نشد (No break)", key="impact_not_break_form")
        
        st.markdown("---")
        
        st.markdown("### ۳. خواص نهایی")
        impact_value = st.number_input(f"خواص ضربه (J/m)", min_value=0.0, key="impact_value_form")
        tensile_value = st.number_input("استحکام کششی (MPa)", min_value=0.0, key="tensile_value_form")
        
        submit_button = st.button(label='💾 ثبت اطلاعات', key="submit_btn")

        if submit_button:
            if df is not None:
                new_data = {
                    "Polymer1_Type": p1_type, "Polymer1_Perc": p1_perc,
                    "Polymer2_Type": p2_type, "Polymer2_Perc": p2_perc,
                    "Polymer3_Type": p3_type, "Polymer3_Perc": p3_perc,
                    "Filler1_Type": f1_type, "Filler1_ParticleSize_um": f1_size, "Filler1_Perc": f1_perc,
                    "Filler2_Type": f2_type, "Filler2_ParticleSize_um": f2_size, "Filler2_Perc": f2_perc,
                    "Additive_Type": a_type, "Additive_Perc": a_perc, "Additive_Functionality": a_func,
                    "Impact_Test_Type": impact_test_type, "Impact_Not_Break": impact_not_break,
                    "Impact_Value_Jm": convert_impact_to_base(impact_value, "J/m"), 
                    "Tensile_Value_MPa": convert_tensile_to_base(tensile_value, "MPa")
                }
                
                new_row = pd.DataFrame([new_data])
                updated_df = pd.concat([df, new_row], ignore_index=True)
                updated_df.to_excel(EXCEL_FILE, index=False)
                st.success("✅ اطلاعات با موفقیت ثبت شد!")
            else:
                st.error("❌ خطا: فایل اکسل دیتاست پیدا نشد. لطفاً آن را آپلود کنید.")

with col_predict:
    st.header("🔮 پیش‌بینی خواص")

    with st.container():
        st.markdown("### مشخصات فرمولاسیون برای پیش‌بینی")

        st.markdown("**پلیمرها**")
        p1_type_p = st.selectbox("نوع پلیمر اول", options=[''] + (unique_values['Polymer1_Type'] if unique_values else []), key="p1_type_p")
        p1_perc_p = st.number_input("درصد پلیمر اول (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="p1_perc_p")
        p2_type_p = st.selectbox("نوع پلیمر دوم", options=[''] + (unique_values['Polymer2_Type'] if unique_values else []), key="p2_type_p")
        p2_perc_p = st.number_input("درصد پلیمر دوم (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="p2_perc_p")
        p3_type_p = st.selectbox("نوع پلیمر سوم", options=[''] + (unique_values['Polymer3_Type'] if unique_values else []), key="p3_type_p")
        p3_perc_p = st.number_input("درصد پلیمر سوم (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="p3_perc_p")
        
        st.markdown("---")

        st.markdown("**فیلرها**")
        f1_type_p = st.selectbox("نوع فیلر اول", options=[''] + (unique_values['Filler1_Type'] if unique_values else []), key="f1_type_p")
        f1_size_p = st.number_input("اندازه ذرات فیلر اول (میکرون)", min_value=0.0, key="f1_size_p")
        f1_perc_p = st.number_input("درصد فیلر اول (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="f1_perc_p")
        f2_type_p = st.selectbox("نوع فیلر دوم", options=[''] + (unique_values['Filler2_Type'] if unique_values else []), key="f2_type_p")
        f2_size_p = st.number_input("اندازه ذرات فیلر دوم (میکرون)", min_value=0.0, key="f2_size_p")
        f2_perc_p = st.number_input("درصد فیلر دوم (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="f2_perc_p")

        st.markdown("---")

        st.markdown("**افزودنی‌ها**")
        a_type_p = st.selectbox("نوع افزودنی", options=[''] + (unique_values['Additive_Type'] if unique_values else []), key="a_type_p")
        a_perc_p = st.number_input("درصد افزودنی (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="a_perc_p")
        a_func_p = st.selectbox("عملکرد افزودنی", options=[''] + ['Toughener', 'Impact Modifier', 'Colorant', 'Antioxidant', 'Unknown'], key="a_func_p")
        
        st.markdown("---")

        st.markdown("### نوع آزمون")
        impact_test_type_p = st.selectbox("نوع آزمون ضربه", options=[''] + ['Charpy', 'Izod', 'Unknown'], key="impact_test_type_p")
        impact_not_break_p = st.checkbox("شکسته نشد (No break)", key="impact_not_break_p")
        
        predict_button = st.button(label='🚀 پیش‌بینی خواص', key="predict_btn")

        if predict_button:
            if impact_model is not None and tensile_model is not None:
                data_to_predict = {
                    'Polymer1_Type': p1_type_p, 'Polymer1_Perc': p1_perc_p,
                    'Polymer2_Type': p2_type_p, 'Polymer2_Perc': p2_perc_p,
                    'Polymer3_Type': p3_type_p, 'Polymer3_Perc': p3_perc_p,
                    'Filler1_Type': f1_type_p, 'Filler1_ParticleSize_um': f1_size_p, 'Filler1_Perc': f1_perc_p,
                    'Filler2_Type': f2_type_p, 'Filler2_ParticleSize_um': f2_size_p, 'Filler2_Perc': f2_perc_p,
                    'Additive_Type': a_type_p, 'Additive_Perc': a_perc_p, 'Additive_Functionality': a_func_p,
                    'Impact_Test_Type': impact_test_type_p, 'Impact_Not_Break': impact_not_break_p
                }
                
                predictions = predict_properties(data_to_predict, impact_model, tensile_model, impact_cols, tensile_cols)

                if predictions:
                    st.success("✅ پیش‌بینی با موفقیت انجام شد!")
                    
                    st.subheader("نتایج پیش‌بینی")
                    # Corrected the display unit to be consistent with the input
                    st.info(f"**خواص ضربه:** {predictions['impact']:.2f} J/m")
                    st.info(f"**استحکام کششی:** {predictions['tensile']:.2f} MPa")
                else:
                    st.error("❌ پیش‌بینی انجام نشد. لطفاً ورودی‌های خود را بررسی کنید.")
            else:
                st.warning("فایل‌های مدل پیدا نشدند. لطفاً آن‌ها را در کنار فایل app.py قرار دهید.")


# --- New section for downloading articles ---
st.markdown("---")
st.header("📄 منابع و مقالات")
st.markdown("می‌توانید مقالات و منابع مربوط به این پروژه را از اینجا دانلود کنید.")

# List of PDF files to be offered for download. 
# You can add more files to this list.
pdf_files = [
    {"name": "مقاله شماره ۱: Mechanical Properties of Blends Containing HDPE and PP", "path": "10.1002@app.1982.070270704.pdf"},
    {"name": "مقاله شماره ۲: Mechanical Properties and Morphologies of Polypropylene With Different Sizes of Calcium Carbonate Particles", "path": "10.1002@pc.20211.pdf"},
    {"name": "مقاله شماره ۲: بررسی اثر پرکننده معدنی تالک بر روی خواص فیزیکی مکانیکی پلی پروپیلن و آلیاژهای آن", "path": "26716-fulltext.pdf"}
]

for file in pdf_files:
    pdf_file_path = file["path"]
    
    try:
        with open(pdf_file_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
        
        st.download_button(
            label=f"دانلود {file['name']}",
            data=pdf_bytes,
            file_name=os.path.basename(pdf_file_path),
            mime="application/pdf"
        )
    except FileNotFoundError:
        st.warning(f"فایل {pdf_file_path} پیدا نشد. لطفاً آن را به پوشه پروژه اضافه کنید.")
