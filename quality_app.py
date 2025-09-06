import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from typing import Dict, Any

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
    html, body {
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
    EXCEL_FILE = 'Polymer_Properties_Processed_by_python1.xlsx'
    
    if not os.path.exists(EXCEL_FILE):
        st.error("فایل اکسل دیتاست پیدا نشد. لطفاً ابتدا آن را آپلود کنید.")
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
def load_tensile_model():
    """
    Loads the trained tensile strength prediction model from a .pkl file.
    """
    MODEL_FILE = 'tensile_model.pkl'
    if not os.path.exists(MODEL_FILE):
        return None
    try:
        model = joblib.load(MODEL_FILE)
        return model
    except Exception as e:
        st.error(f"خطا در بارگذاری مدل پیش‌بینی: {e}")
        return None

def find_best_match(new_data, df):
    """
    Finds the best matching formulation in the dataset based on the new input,
    using a simple Euclidean distance.
    """
    if df is None or df.empty:
        return None, None, None

    # Get numerical data for comparison
    df_numeric = df[['Polymer1_Perc', 'Polymer2_Perc', 'Polymer3_Perc',
                     'Filler1_Perc', 'Filler1_ParticleSize_um', 'Filler2_Perc',
                     'Filler2_ParticleSize_um', 'Additive_Perc']]
    
    new_data_numeric = np.array([new_data['Polymer1_Perc'], new_data['Polymer2_Perc'], new_data['Polymer3_Perc'],
                                 new_data['Filler1_Perc'], new_data['Filler1_ParticleSize_um'], new_data['Filler2_Perc'],
                                 new_data['Filler2_ParticleSize_um'], new_data['Additive_Perc']])

    # Calculate Euclidean distance
    distances = np.linalg.norm(df_numeric.values - new_data_numeric, axis=1)
    
    best_match_index = np.argmin(distances)
    best_match = df.iloc[best_match_index]
    
    return best_match, best_match_index, distances[best_match_index]


# --- Main App Structure ---
st.set_page_config(layout="wide", page_title="کنترل کیفیت پلیمر")
st.markdown("<div class='centered-title'><h1>🧪 برنامه کنترل کیفیت کامپوزیت‌های پلیمری</h1></div>", unsafe_allow_html=True)
st.markdown(
    """
    <div class='centered-description'>
    <p>این برنامه به شما امکان می‌دهد خواص فرمولاسیون‌های پلیمری را ثبت کنید، آن‌ها را با فرمولاسیون‌های موجود مقایسه کرده و بهترین فرمولاسیون‌های مشابه را پیدا کنید.</p>
    </div>
    """, unsafe_allow_html=True)


df, unique_values = load_data_and_get_unique_values()
model = load_tensile_model()


# --- File Uploader and Data Display ---
st.header("📂 آپلود و مشاهده دیتاست")

uploaded_file = st.file_uploader("فایل اکسل (Excel) خود را آپلود کنید", type=['xlsx', 'xls'])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        df.to_excel('Polymer_Properties_Processed_by_python1.xlsx', index=False)
        st.success("✅ فایل با موفقیت آپلود شد! برنامه در حال به‌روزرسانی است.")
        st.dataframe(df, use_container_width=True)
        # Rerun the app to load the new data
        st.experimental_rerun()
    except Exception as e:
        st.error(f"خطا در خواندن فایل آپلود شده: {e}")
        
if df is not None:
    st.subheader("پیش‌نمایش دیتاست موجود")
    st.dataframe(df, use_container_width=True)

# --- Prediction and Suggestion Section ---
st.markdown("---")
st.header("🔮 پیش‌بینی خواص و ارائه پیشنهاد")
if model is None:
    st.warning("فایل مدل پیش‌بینی (tensile_model.pkl) پیدا نشد. لطفاً ابتدا مدل را آموزش داده و آپلود کنید.")
elif unique_values:
    # Use the same input fields for prediction
    with st.form(key='prediction_form'):
        st.markdown("**مشخصات فرمولاسیون برای پیش‌بینی**")
        col3, col4 = st.columns(2)
        with col3:
            p1_type_pred = st.selectbox("نوع پلیمر اول", options=[''] + list(unique_values['Polymer1_Type']), key="p1_type_pred")
            p1_perc_pred = st.number_input("درصد پلیمر اول (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="p1_perc_pred")
            p2_type_pred = st.selectbox("نوع پلیمر دوم", options=[''] + list(unique_values['Polymer2_Type']), key="p2_type_pred")
            p2_perc_pred = st.number_input("درصد پلیمر دوم (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="p2_perc_pred")
            p3_type_pred = st.selectbox("نوع پلیمر سوم", options=[''] + list(unique_values['Polymer3_Type']), key="p3_type_pred")
            p3_perc_pred = st.number_input("درصد پلیمر سوم (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="p3_perc_pred")
            f1_type_pred = st.selectbox("نوع فیلر اول", options=[''] + list(unique_values['Filler1_Type']), key="f1_type_pred")
            f1_size_pred = st.number_input("اندازه ذرات فیلر اول (میکرون)", min_value=0.0, key="f1_size_pred")
            f1_perc_pred = st.number_input("درصد فیلر اول (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="f1_perc_pred")
            f2_type_pred = st.selectbox("نوع فیلر دوم", options=[''] + list(unique_values['Filler2_Type']), key="f2_type_pred")
            f2_size_pred = st.number_input("اندازه ذرات فیلر دوم (میکرون)", min_value=0.0, key="f2_size_pred")
            f2_perc_pred = st.number_input("درصد فیلر دوم (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="f2_perc_pred")

        with col4:
            a_type_pred = st.selectbox("نوع افزودنی", options=[''] + list(unique_values['Additive_Type']), key="a_type_pred")
            a_perc_pred = st.number_input("درصد افزودنی (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="a_perc_pred")
            a_func_pred = st.selectbox("عملکرد افزودنی", options=[''] + ['Toughener', 'Impact Modifier', 'Colorant', 'Antioxidant', 'Unknown'], key="a_func_pred")
            impact_test_type_pred = st.selectbox("نوع آزمون ضربه", options=[''] + ['Charpy', 'Izod'], key="impact_test_type_pred")
            target_tensile = st.number_input("مقاومت کششی هدف (MPa)", min_value=0.0, key="target_tensile")
            target_impact = st.number_input("مقاومت ضربه هدف (J/m)", min_value=0.0, key="target_impact")

        predict_button = st.form_submit_button(label='🔮 پیش‌بینی خواص')

    if predict_button:
        # Create a DataFrame for prediction, similar to the training script
        prediction_data = pd.DataFrame([{
            'Polymer1_Type': p1_type_pred, 'Polymer1_Perc': p1_perc_pred,
            'Polymer2_Type': p2_type_pred, 'Polymer2_Perc': p2_perc_pred,
            'Polymer3_Type': p3_type_pred, 'Polymer3_Perc': p3_perc_pred,
            'Filler1_Type': f1_type_pred, 'Filler1_ParticleSize_um': f1_size_pred, 'Filler1_Perc': f1_perc_pred,
            'Filler2_Type': f2_type_pred, 'Filler2_ParticleSize_um': f2_size_pred, 'Filler2_Perc': f2_perc_pred,
            'Additive_Type': a_type_pred, 'Additive_Perc': a_perc_pred, 'Additive_Functionality': a_func_pred,
            'Impact_Test_Type': impact_test_type_pred, 'Tensile_Strength': np.nan # This is the target we are predicting
        }])
        
        # Define categorical features used during training
        categorical_features = ['Polymer1_Type', 'Polymer2_Type', 'Polymer3_Type',
                                'Filler1_Type', 'Filler2_Type', 'Additive_Type',
                                'Impact_Test_Type']

        # Get all unique values for each categorical feature from the entire dataset
        # This is crucial to ensure all possible columns are created during one-hot encoding
        all_unique_values = {col: sorted(df[col].unique()) for col in categorical_features if col in df.columns}
        
        # One-Hot Encoding and aligning columns with the trained model
        for col in categorical_features:
            if col in prediction_data.columns:
                prediction_data = pd.get_dummies(prediction_data, columns=[col], drop_first=True)
        
        # Get the columns from the original training dataframe (or a complete dummy df)
        # to ensure alignment. This is a robust way to handle unseen categories.
        dummy_df = pd.get_dummies(df[categorical_features], columns=categorical_features, drop_first=True)
        
        missing_cols = set(dummy_df.columns) - set(prediction_data.columns)
        for c in missing_cols:
            prediction_data[c] = 0

        # Align columns
        X_pred = prediction_data[dummy_df.columns]

        try:
            # Predict the tensile strength
            predicted_tensile = model.predict(X_pred)[0]
            
            st.markdown(f"**مقاومت کششی پیش‌بینی‌شده:** **{predicted_tensile:.2f} MPa**")
            
            # Compare with target
            if target_tensile > 0:
                if predicted_tensile >= target_tensile:
                    st.success(f"✅ نتیجه پیش‌بینی‌شده ({predicted_tensile:.2f} MPa) بیشتر یا برابر با مقدار هدف شما ({target_tensile} MPa) است. فرمولاسیون شما مناسب است!")
                else:
                    st.warning(f"⚠️ نتیجه پیش‌بینی‌شده ({predicted_tensile:.2f} MPa) کمتر از مقدار هدف شما ({target_tensile} MPa) است. برای رسیدن به خواص مطلوب، نیاز به بهینه‌سازی دارید.")
        except Exception as e:
            st.error(f"❌ خطایی در پیش‌بینی رخ داد: {e}")
