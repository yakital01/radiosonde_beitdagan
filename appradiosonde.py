import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pdfplumber
import re

# הגדרת תצורת העמוד ב-Streamlit
st.set_page_config(
    page_title="ניתוח פרופיל רדיוסונדה - בית דגן",
    page_icon="🎈",
    layout="centered"
)

st.title("📊 ניתוח פרופיל רדיוסונדה")
st.write("העלה קובץ PDF של מדידת רדיוסונדה מבית דגן לקבלת פרופילי $N$ ו-$M$.")

# פונקציה למזעור ועיבוד נתוני ה-PDF
def process_radiosonde_pdf(pdf_file):
    lines = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                lines.extend(text.split('\n'))
    
    data = []
    # חילוץ שורות נתונים נומריות
    for line in lines:
        parts = line.strip().split()
        # בדיקה האם השורה מכילה נתונים מספריים של סונדה
        if len(parts) >= 6:
            try:
                p = float(parts[0])      # לחץ [hPa]
                h = float(parts[1])      # גובה [m]
                t = float(parts[2])      # טמפרטורה [C]
                rh = float(parts[3])     # לחות יחסית [%]
                
                # חישוב לחץ אדים רווי e_s [hPa]
                e_s = 6.112 * np.exp((17.67 * t) / (t + 243.5))
                # חישוב לחץ אדים בפועל e [hPa]
                e = (rh / 100.0) * e_s
                # טמפרטורה במעלות קלווין T [K]
                T_k = t + 273.15
                
                # חישוב Refractivity (N)
                N = (77.6 * (p / T_k)) + (3.73e5 * (e / (T_k**2)))
                
                # חישוב Modified Refractivity (M)
                M = N + (h / 0.157)
                
                data.append({
                    'Pressure_hPa': p,
                    'Height_m': h,
                    'Temp_C': t,
                    'RH_pct': rh,
                    'N': N,
                    'M': M
                })
            except ValueError:
                continue

    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values('Height_m').reset_index(drop=True)
    return df

# רכיב העלאת הקבצים
uploaded_file = st.file_uploader("בחר קובץ PDF של רדיוסונדה", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("מעבד את הנתונים..."):
        df = process_radiosonde_pdf(uploaded_file)
    
    if df.empty:
        st.error("לא ניתן היה לחלץ נתונים מתוך קובץ ה-PDF. ודא שזהו קובץ תקין מבית דגן.")
    else:
        st.success("הקובץ עובד בהצלחה!")
        
        # סליידר לסינון גובה מקסימלי
        max_h_val = int(df['Height_m'].max())
        selected_max_h = st.slider("בחר גובה מקסימלי לתצוגה (מטרים):", 0, max_h_val, min(3000, max_h_val), step=100)
        
        # סינון ה-DataFrame לפי הגובה שנבחר
        df_filtered = df[df['Height_m'] <= selected_max_h]
        
        # --- יצירת גרף M (Modified Refractivity) ---
        fig_M = go.Figure()
        fig_M.add_trace(go.Scatter(
            x=df_filtered['M'], 
            y=df_filtered['Height_m'],
            mode='lines+markers',
            name='M Profile',
            line=dict(color='red', width=2),
            marker=dict(size=4)
        ))
        fig_M.update_layout(
            title="פרופיל M (Modified Refractivity)",
            xaxis_title="M [M-units]",
            yaxis_title="גובה [מטרים]",
            height=430,
            margin=dict(l=20, r=20, t=40, b=20),
            dragmode=False
        )
        fig_M.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig_M.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        # --- יצירת גרף N (Refractivity) ---
        fig_N = go.Figure()
        fig_N.add_trace(go.Scatter(
            x=df_filtered['N'], 
            y=df_filtered['Height_m'],
            mode='lines+markers',
            name='N Profile',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        fig_N.update_layout(
            title="פרופיל N (Refractivity)",
            xaxis_title="N [N-units]",
            yaxis_title="גובה [מטרים]",
            height=430,
            margin=dict(l=20, r=20, t=40, b=20),
            dragmode=False
        )
        fig_N.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig_N.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

        # הגדרות מניעת זום ומגע מיותר בטלפון
        mobile_config = {
            'scrollZoom': False,
            'displayModeBar': False
        }

        # --- תצוגת הטאבים במובייל (M מופיע ראשון, N שני) ---
        tab_M, tab_N = st.tabs(["$M$ - Refractivity", "$N$ - Modified Refractivity"])

        with tab_M:
            st.plotly_chart(fig_M, use_container_width=True, config=mobile_config)

        with tab_N:
            st.plotly_chart(fig_N, use_container_width=True, config=mobile_config)

        # --- טבלת נתונים מעובדת ---
        with st.expander("הצג טבלת נתונים מעובדת"):
            st.dataframe(df_filtered)
