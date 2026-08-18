import numpy as np
import pandas as pd
import pdfplumber
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# הגדרת תצורת הדף במובייל
st.set_page_config(
    page_title="ניתוח רדיוסונדה", layout="wide", initial_sidebar_state="collapsed"
)

st.title("📊 ניתוח פרופילי רדיוסונדה (בית דגן)")
st.write(
    "העלה קובץ PDF של תצפית רום לקבלת פרופילי N ו-M כתלות בגובה AMSL (z)."
)

# רכיב העלאת קובץ
uploaded_file = st.file_uploader(
    "בחר קובץ PDF (sonde_*.pdf)", type=["pdf"]
)

if uploaded_file is not None:
    rows = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue

            for line in text.split("\n"):
                parts = line.split()
                if not parts:
                    continue

                # טיפול בשורת SURFACE
                if "SURFACE" in parts:
                    try:
                        pres = float(parts[0])
                        height = 34.0  # גובה פני השטח בבית דגן
                        temp = float(parts[3])
                        rh = float(parts[4])
                        rows.append([pres, height, temp, rh])
                    except (ValueError, IndexError):
                        pass

                # טיפול בשורות מדידה רגילות
                elif parts[0].isdigit() and len(parts) >= 6:
                    try:
                        pres = float(parts[0])
                        height = float(parts[2])
                        temp = float(parts[4])
                        rh = float(parts[5])
                        rows.append([pres, height, temp, rh])
                    except (ValueError, IndexError):
                        pass

    if rows:
        df = pd.DataFrame(
            rows, columns=["Pressure_hPa", "Height_m", "Temp_C", "RH"]
        )
        df = df.sort_values(by="Height_m").reset_index(drop=True)

        # חישוב N ו-M
        T_K = df["Temp_C"] + 273.15
        e_s = 6.112 * np.exp((17.67 * df["Temp_C"]) / (df["Temp_C"] + 243.5))
        e = (df["RH"] / 100.0) * e_s

        df["N"] = (77.6 * (df["Pressure_hPa"] / T_K)) + (
            3.73e5 * (e / (T_K**2))
        )
        df["M"] = df["N"] + (0.157 * df["Height_m"])

        # סינון עד גובה 3000 מטר לצפייה אופטימלית
        max_h = st.slider("גובה מקסימלי לתצוגה (מטרים):", 500, 10000, 3000, 500)
        df_filtered = df[df["Height_m"] <= max_h]

        # יצירת גרפים אינטראקטיביים עם Plotly (מותאם למגע במובייל)
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("N Profile (Refractivity)", "M Profile (Modified)"),
            shared_yaxes=True,
        )

        fig.add_trace(
            go.Scatter(
                x=df_filtered["N"],
                y=df_filtered["Height_m"],
                mode="lines+markers",
                name="N",
                line=dict(color="blue"),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df_filtered["M"],
                y=df_filtered["Height_m"],
                mode="lines+markers",
                name="M",
                line=dict(color="red"),
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(title_text="N [N-units]", row=1, col=1)
        fig.update_xaxes(title_text="M [M-units]", row=1, col=2)
        fig.update_yaxes(title_text="Height AMSL z [m]", row=1, col=1)

       # 1. הגדרת גובה מתאים וביטול זום מציק במגע אצבע
        fig.update_layout(
        height=420,  # גובה מותאם שלא ימתח את הגרף
        margin=dict(l=20, r=20, t=40, b=20),  # שוליים קטנים יותר לניצול מסך הטלפון
        dragmode=False,  # מבטל זום/הזזה אקראית של הגרף במגע אצבע
        )

        # הצגת קווי רשת גם בציר אופקי (גובה) וגם בציר אנכי (ערכים)
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

        # 2. הצגת הגרף ב-Streamlit עם הגדרת ניתוק אינטראקציית מגע
        config = {
        'scrollZoom': False,  # מנעת זום בגלילה/מגע
        'displayModeBar': False  # הסתרת סרגל הכלים העליון של Plotly שתופס מקום בטלפון
        }

        st.plotly_chart(fig, use_container_width=True, config=config)

        # טבלת נתונים
        with st.expander("הצג טבלת נתונים מעובדת"):
            st.dataframe(df)
    else:
        st.error("לא נשלפו נתונים תקינים מקובץ ה-PDF.")
