import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pdfplumber

st.set_page_config(
    page_title="ניתוח פרופיל רדיוסונדה - בית דגן",
    page_icon="🎈",
    layout="centered"
)

st.title("📊 ניתוח פרופיל רדיוסונדה")
st.write("העלה קובץ PDF של מדידת רדיוסונדה מבית דגן לקבלת פרופילי $N$ ו-$M$.")

# פונקציית העיבוד המקורית והתקינה שלך
def process_radiosonde_pdf(pdf_file):
    lines = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                lines.extend(text.split('\n'))
    
    data = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 4:
            try:
                p = float(parts[0])      # לחץ [hPa]
                h = float(parts[1])      # גובה [m]
                t = float(parts[2])      # טמפרטורה [C]
                rh = float(parts[3])     # לחות יחסית [%]
                
                if p > 0 and 0 <= rh <= 100:
                    e_s = 6.112 * np.exp((17.67 * t) / (t + 243.5))
                    e = (rh / 100.0) * e_s
                    T_k = t + 273.15
                    
                    # חישוב N ו-M
                    N = (77.6 * (p / T_k)) + (3.73e5 * (e / (T_k**2)))
                    M = N + (0.157 * h)
                    
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
        df = df.drop_duplicates(subset=['Height_m']).sort_values('Height_m').reset_index(drop=True)
    return df

uploaded_file = st.file_uploader("בחר קובץ PDF של רדיוסונדה", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("מעבד את הנתונים..."):
        df = process_radiosonde_pdf(uploaded_file)
    
    if df.empty:
        st.error("לא ניתן היה לחלץ נתונים מתוך קובץ ה-PDF. ודא שזהו קובץ תקין מבית דגן.")
    else:
        st.success("הקובץ עובד בהצלחה!")
        
        max_h_val = int(df['Height_m'].max())
        selected_max_h = st.slider("בחר גובה מקסימלי לתצוגה (מטרים):", 0, max_h_val, min(3000, max_h_val), step=100)
        
        df_filtered = df[df['Height_m'] <= selected_max_h]
        
        # --- גרף M (Modified Refractivity) ---
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
            dragmode=False  # מונע זום אקראי במגע אצבע במובייל
        )
        fig_M.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', autorange=True)
        fig_M.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        # --- גרף N (Refractivity) ---
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
            dragmode=False  # מונע זום אקראי במגע אצבע במובייל
        )
        fig_N.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', autorange=True)
        fig_N.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

        # הגדרות מובייל
        mobile_config = {
            'scrollZoom': False,
            'displayModeBar': False
        }

        # --- תצוגת טאבים: M ראשון, N שני ---
        tab_M, tab_N = st.tabs(["$M$ - Refractivity", "$N$ - Modified Refractivity"])

        with tab_M:
            st.plotly_chart(fig_M, use_container_width=True, config=mobile_config)

        with tab_N:
            st.plotly_chart(fig_N, use_container_width=True, config=mobile_config)

        # --- טבלת נתונים ---
        with st.expander("הצג טבלת נתונים מעובדת"):
            st.dataframe(df_filtered)
