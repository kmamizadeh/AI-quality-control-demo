import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# --- Paths to the files. These should be relative to the script location.
EXCEL_FILE = 'Polymer_Properties_Processed_by_python1.xlsx'
IMPACT_MODEL_FILE = 'regression_model.pkl'
TENSILE_MODEL_FILE = 'tensile_model.pkl'

# --- Custom CSS for a modern, clean look ---
st.markdown("""
<style>
    /* General body and typography */
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

</style>
""", unsafe_allow_html=True)


# --- Data and Model Loading (with caching) ---
@st.cache_data
def load_data_and_get_unique_values():
    if not os.path.exists(EXCEL_FILE):
        return None, None

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


@st.cache_resource
def load_model_and_get_columns():
    try:
        if not os.path.exists(IMPACT_MODEL_FILE) or not os.path.exists(TENSILE_MODEL_FILE):
            st.error("فایل های مدل (.pkl) پیدا نشدند. لطفا مطمئن شوید که در مسیر صحیح قرار دارند.")
            return None, None, None, None

        impact_model = joblib.load(IMPACT_MODEL_FILE)
        tensile_model = joblib.load(TENSILE_MODEL_FILE)

        impact_model_columns = impact_model.feature_names_in_.tolist()
        tensile_model_columns = tensile_model.feature_names_in_.tolist()

        return impact_model, tensile_model, impact_model_columns, tensile_model_columns

    except FileNotFoundError:
        st.error("خطا: فایل‌های مدل پیدا نشدند.")
        return None, None, None, None
    except Exception as e:
        st.error(f"خطا در بارگذاری مدل: {e}")
        return None, None, None, None


def predict_properties(data_to_predict, impact_model, tensile_model, impact_cols, tensile_cols):
    try:
        df_impact = pd.DataFrame(columns=impact_cols)
        df_impact.loc[0] = 0
        df_tensile = pd.DataFrame(columns=tensile_cols)
        df_tensile.loc[0] = 0

        for key, value in data_to_predict.items():
            if value is None:
                continue
            if key in df_impact.columns:
                df_impact.loc[0, key] = value
            if key in df_tensile.columns:
                df_tensile.loc[0, key] = value
            if isinstance(value, str):
                if f'{key}_{value}' in df_impact.columns:
                    df_impact.loc[0, f'{key}_{value}'] = 1
                if f'{key}_{value}' in df_tensile.columns:
                    df_tensile.loc[0, f'{key}_{value}'] = 1

        categorical_cols = ['Polymer1_Type', 'Polymer2_Type', 'Polymer3_Type',
                            'Filler1_Type', 'Filler2_Type', 'Additive_Type',
                            'Impact_Test_Type']
        for col in categorical_cols:
            if col in df_impact.columns:
                df_impact = df_impact.drop(columns=[col])
            if col in df_tensile.columns:
                df_tensile = df_tensile.drop(columns=[col])

        impact_pred = impact_model.predict(df_impact)[0]
        tensile_pred = tensile_model.predict(df_tensile)[0]

        return {'impact': impact_pred, 'tensile': tensile_pred}
    except Exception as e:
        st.error(f"خطا در پیش‌بینی: {e}")
        return None


# --- Main App Structure ---
st.set_page_config(layout="wide", page_title="AI Quality Control Dashboard")
st.title("داشبورد کنترل کیفیت هوشمند")
st.markdown(
    """
    این برنامه با استفاده از مدل‌های هوش مصنوعی، خواص یک فرمولاسیون را پیش‌بینی کرده
    و آن‌ها را با استانداردهای کیفی مورد انتظار مقایسه می‌کند.
    """
)

df, unique_values = load_data_and_get_unique_values()
impact_model, tensile_model, impact_cols, tensile_cols = load_model_and_get_columns()

col_inputs, col_dashboard = st.columns([1, 1])

# --- Input Section (left column) ---
with col_inputs:
    st.header("۱. تعریف فرمولاسیون و تارگت‌های کیفی")
    st.markdown("---")

    with st.container():
        st.subheader("مشخصات فرمولاسیون")

        st.markdown("**پلیمرها**")
        p1_type_p = st.selectbox("نوع پلیمر اول", options=[''] + unique_values['Polymer1_Type'], key="p1_type_p")
        p1_perc_p = st.number_input("درصد پلیمر اول (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1,
                                    key="p1_perc_p")
        p2_type_p = st.selectbox("نوع پلیمر دوم", options=[''] + unique_values['Polymer2_Type'], key="p2_type_p")
        p2_perc_p = st.number_input("درصد پلیمر دوم (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1,
                                    key="p2_perc_p")
        p3_type_p = st.selectbox("نوع پلیمر سوم", options=[''] + unique_values['Polymer3_Type'], key="p3_type_p")
        p3_perc_p = st.number_input("درصد پلیمر سوم (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1,
                                    key="p3_perc_p")

        st.markdown("---")

        st.markdown("**فیلرها**")
        f1_type_p = st.selectbox("نوع فیلر اول", options=[''] + unique_values['Filler1_Type'], key="f1_type_p")
        f1_size_p = st.number_input("اندازه ذرات فیلر اول (میکرون)", min_value=0.0, key="f1_size_p")
        f1_perc_p = st.number_input("درصد فیلر اول (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1,
                                    key="f1_perc_p")
        f2_type_p = st.selectbox("نوع فیلر دوم", options=[''] + unique_values['Filler2_Type'], key="f2_type_p")
        f2_size_p = st.number_input("اندازه ذرات فیلر دوم (میکرون)", min_value=0.0, key="f2_size_p")
        f2_perc_p = st.number_input("درصد فیلر دوم (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1,
                                    key="f2_perc_p")

        st.markdown("---")

        st.markdown("**افزودنی‌ها**")
        a_type_p = st.selectbox("نوع افزودنی", options=[''] + unique_values['Additive_Type'], key="a_type_p")
        a_perc_p = st.number_input("درصد افزودنی (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1,
                                   key="a_perc_p")
        a_func_p = st.selectbox("عملکرد افزودنی",
                                options=[''] + ['Toughener', 'Impact Modifier', 'Colorant', 'Antioxidant', 'Unknown'],
                                key="a_func_p")

    st.markdown("---")

    with st.container():
        st.subheader("تارگت‌های کیفی")
        st.info("لطفاً حداقل خواص مورد انتظار برای محصول را وارد کنید.")
        target_impact = st.number_input("حداقل خواص ضربه (J/m²)", min_value=0.0, step=0.1, key="target_impact")
        target_tensile = st.number_input("حداقل استحکام کششی (MPa)", min_value=0.0, step=0.1, key="target_tensile")

    st.markdown("---")

    predict_button = st.button("پیش‌بینی و اجرای کنترل کیفیت", use_container_width=True, type="primary")

# --- Dashboard Logic (right column) ---
with col_dashboard:
    if predict_button:
        # Check if models are loaded
        if impact_model is None or tensile_model is None:
            st.warning("فایل‌های مدل پیدا نشدند. لطفاً آن‌ها را در کنار فایل برنامه قرار دهید.")
        else:
            data_to_predict = {
                'Polymer1_Type': p1_type_p, 'Polymer1_Perc': p1_perc_p,
                'Polymer2_Type': p2_type_p, 'Polymer2_Perc': p2_perc_p,
                'Polymer3_Type': p3_type_p, 'Polymer3_Perc': p3_perc_p,
                'Filler1_Type': f1_type_p, 'Filler1_ParticleSize_um': f1_size_p, 'Filler1_Perc': f1_perc_p,
                'Filler2_Type': f2_type_p, 'Filler2_ParticleSize_um': f2_size_p, 'Filler2_Perc': f2_perc_p,
                'Additive_Type': a_type_p, 'Additive_Perc': a_perc_p, 'Additive_Functionality': a_func_p,
                'Impact_Test_Type': 'Unknown'  # The model for tensile and impact prediction does not need this feature
            }

            # Perform prediction
            predictions = predict_properties(data_to_predict, impact_model, tensile_model, impact_cols, tensile_cols)

            if predictions:
                st.header("۲. داشبورد کنترل کیفیت")
                st.markdown("---")

                # Perform the quality checks
                passed_impact = predictions['impact'] >= target_impact
                passed_tensile = predictions['tensile'] >= target_tensile

                # Display overall verdict
                if passed_impact and passed_tensile:
                    st.success("✅ نتیجه: تأیید شده", icon="🎉")
                    st.write("تمام خواص پیش‌بینی شده با تارگت‌های مورد نظر شما مطابقت دارند.")
                else:
                    st.error("❌ نتیجه: رد شده", icon="🚨")
                    st.write("خواص پیش‌بینی شده به تارگت‌های مورد انتظار نرسیدند.")

                st.markdown("---")
                st.subheader("مقایسه تفصیلی")

                # Display metrics for each property
                col_impact, col_tensile = st.columns(2)

                with col_impact:
                    st.metric(
                        label=f"خواص ضربه (تارگت: {target_impact:.2f} J/m²)",
                        value=f"{predictions['impact']:.2f} J/m²",
                        delta=f"{predictions['impact'] - target_impact:.2f} J/m²"
                    )

                with col_tensile:
                    st.metric(
                        label=f"استحکام کششی (تارگت: {target_tensile:.2f} MPa)",
                        value=f"{predictions['tensile']:.2f} MPa",
                        delta=f"{predictions['tensile'] - target_tensile:.2f} MPa"
                    )

                st.markdown("---")
                st.subheader("گزارش هوشمند")

                if not passed_impact:
                    st.warning("⚠️ **خواص ضربه:** مقدار پیش‌بینی شده از تارگت شما کمتر است.")
                    st.info("💡 پیشنهاد: برای افزایش خواص ضربه، می‌توانید درصد افزودنی‌های 'Toughener' را افزایش دهید.")

                if not passed_tensile:
                    st.warning("⚠️ **استحکام کششی:** مقدار پیش‌بینی شده از تارگت شما کمتر است.")
                    st.info(
                        "💡 پیشنهاد: برای افزایش استحکام کششی، می‌توانید درصد فیلرها را افزایش دهید یا از یک پلیمر با مدول الاستیک بالاتر استفاده کنید.")
            else:
                st.error("❌ پیش‌بینی انجام نشد. لطفاً ورودی‌های خود را بررسی کنید.")

