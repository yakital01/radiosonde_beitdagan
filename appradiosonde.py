import numpy as np
import pandas as pd
import pdfplumber
import plotly.graph_objects as go
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

        # --- חישוב גבולות ציר X המותאמים (מכפלה של 50 כלפי מטה) ---
        # עבור M:
        m_min_val = df_filtered["M"].min()
        m_start_tick = np.floor(m_min_val / 50.0) * 50.0
        m_max_val = df_filtered["M"].max() + 10  # מרווח קל מימין

        # עבור N:
        n_min_val = df_filtered["N"].min()
        n_start_tick = np.floor(n_min_val / 50.0) * 50.0
        n_max_val = df_filtered["N"].max() + 10  # מרווח קל מימין

        # --- יצירת גרף M (Modified Refractivity) ---
        fig_M = go.Figure()
        fig_M.add_trace(
            go.Scatter(
                x=df_filtered["M"],
                y=df_filtered["Height_m"],
                mode="lines+markers",
                name="M",
                line=dict(color="red"),
            )
        )
        fig_M.update_layout(
            title="M Profile (Modified Refractivity)",
            xaxis_title="M [M-units]",
            yaxis_title="Height AMSL z [m]",
            height=450,
            margin=dict(l=35, r=35, t=40, b=40),
            dragmode=False
        )
        fig_M.update_xaxes(
            showgrid=True, 
            gridwidth=1, 
            gridcolor='LightGray',
            ticks="outside",
            showline=True,
            linecolor='black',
            tick0=m_start_tick,       # השנתה הראשונה בדיוק במכפלה של 50 מתחת למינימום
            dtick=50,                 # מרווח קבוע של 50 יחידות בין שנתה לשנתה
            range=[m_start_tick - 5, m_max_val]  # הגדרת טווח הציר
        )
        fig_M.update_yaxes(
            showgrid=True, 
            gridwidth=1, 
            gridcolor='LightGray',
            showline=True,
            linecolor='black'
        )

        # --- יצירת גרף N (Refractivity) ---
        fig_N = go.Figure()
        fig_N.add_trace(
            go.Scatter(
                x=df_filtered["N"],
                y=df_filtered["Height_m"],
                mode="lines+markers",
                name="N",
                line=dict(color="blue"),
            )
        )
        fig_N.update_layout(
            title="N Profile (Refractivity)",
            xaxis_title="N [N-units]",
            yaxis_title="Height AMSL z [m]",
            height=450,
            margin=dict(l=35, r=35, t=40, b=40),
            dragmode=False
        )
        fig_N.update_xaxes(
            showgrid=True, 
            gridwidth=1, 
            gridcolor='LightGray',
            ticks="outside",
            showline=True,
            linecolor='black',
            tick0=n_start_tick,       # השנתה הראשונה בדיוק במכפלה של 50 מתחת למינימום
            dtick=50,                 # מרווח קבוע של 50 יחידות
            range=[n_start_tick - 5, n_max_val]
        )
        fig_N.update_yaxes(
            showgrid=True, 
            gridwidth=1, 
            gridcolor='LightGray',
            showline=True,
            linecolor='black'
        )

        # הגדרות תצוגת מובייל
        mobile_config = {
            'scrollZoom': False,
            'displayModeBar': False
        }

        # --- הצגה בטאבים ---
        tab_M, tab_N = st.tabs(["$M$ - Modified Refractivity", "$N$ - Refractivity"])

        with tab_M:
            st.plotly_chart(fig_M, use_container_width=True, config=mobile_config)

        with tab_N:
            st.plotly_chart(fig_N, use_container_width=True, config=mobile_config)

        # טבלת נתונים
        with st.expander("הצג טבלת נתונים מעובדת"):
            st.dataframe(df)
    else:
        st.error("לא נשלפו נתונים תקינים מקובץ ה-PDF.")
