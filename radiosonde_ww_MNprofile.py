import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import re

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Atmospheric Refractivity & Duct Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Helper Functions: Calculation & Detection
# ---------------------------------------------------------

def calculate_refractivity(df):
    """
    Calculates Refractivity N and Modified Refractivity M.
    P : Pressure [hPa]
    T : Temperature [C]
    TD: Dewpoint Temperature [C]
    HGHT: Height above sea level [m]
    """
    # Convert T to Kelvin
    T_k = df['T'] + 273.15
    
    # Vapor Pressure e [hPa] using Magnus formula
    e = 6.112 * np.exp((17.67 * df['TD']) / (df['TD'] + 243.5))
    
    # Refractivity N
    df['N'] = (77.6 * df['P'] / T_k) + (3.73e5 * e / (T_k**2))
    
    # Modified Refractivity M
    # M = N + 0.157 * z (where z is height in meters)
    df['M'] = df['N'] + 0.157 * df['HGHT']
    
    return df

def detect_ducts(z_arr, m_arr, min_delta_m=5.0, min_delta_z=40.0, c_speed=3e8):
    """
    Detects Atmospheric Ducts (Surface & Elevated).
    Fixes edge-case where duct starts directly at the ground level (i=0).
    Outputs results in English.
    """
    z = np.array(z_arr, dtype=float)
    m = np.array(m_arr, dtype=float)
    
    ducts = []
    n = len(z)
    i = 0
    
    while i < n - 1:
        # Check for negative M gradient (dM/dz < 0)
        if m[i+1] < m[i]:
            base_idx = i
            base_z = z[base_idx]
            base_m = m[base_idx]
            
            # Trace down to local minimum of M
            j = i + 1
            while j < n - 1 and m[j+1] < m[j]:
                j += 1
            
            top_idx = j
            top_z = z[top_idx]
            top_m = m[top_idx]
            
            delta_m = base_m - top_m
            layer_thickness = top_z - base_z
            
            # Apply user-defined thresholds
            if delta_m >= min_delta_m and layer_thickness >= min_delta_z:
                
                # Physical Classification
                if base_idx == 0:
                    duct_type = "Surface Duct"
                    d_eff = layer_thickness
                else:
                    duct_type = "Elevated Duct"
                    d_eff = layer_thickness
                
                # Cutoff Frequency Calculation
                # fc = (2/3) * c / (d_eff * sqrt(2 * 10^-6 * delta_M))
                if delta_m > 0 and d_eff > 0:
                    fc_hz = (2.0 / 3.0) * c_speed / (d_eff * np.sqrt(2 * 1e-6 * delta_m))
                    fc_ghz = fc_hz / 1e9
                else:
                    fc_ghz = np.nan
                
                ducts.append({
                    "Duct Type": duct_type,
                    "Base Alt [m]": round(base_z, 1),
                    "Top Alt [m]": round(top_z, 1),
                    "Thickness [m]": round(layer_thickness, 1),
                    "Delta M": round(delta_m, 2),
                    "Cutoff Freq [GHz]": round(fc_ghz, 3) if not np.isnan(fc_ghz) else "N/A"
                })
            
            i = top_idx
        else:
            i += 1

    if not ducts:
        return None
    
    return pd.DataFrame(ducts)

@st.cache_data(ttl=3600)
def fetch_uwyo_sounding(station_id, date_str):
    """
    Fetches raw text sounding data from University of Wyoming.
    """
    url = f"https://weather.uwyo.edu/wsgi/sounding?datetime={date_str}&id={station_id}&src=FM35&type=TEXT:LIST"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None, f"HTTP Error {response.status_code}"
        
        soup = BeautifulSoup(response.text, 'html.parser')
        pre_tag = soup.find('pre')
        
        if not pre_tag:
            return None, "No data found for the selected station/date."
        
        lines = pre_tag.text.split('\n')
        
        # Find data start index
        data_start = -1
        for idx, line in enumerate(lines):
            if "PRES" in line and "HGHT" in line:
                data_start = idx + 3
                break
                
        if data_start == -1:
            return None, "Failed to parse sounding headers."
            
        data_rows = []
        for line in lines[data_start:]:
            if line.strip() == "" or "Station information" in line:
                break
            parts = line.split()
            if len(parts) >= 11:
                try:
                    pres = float(parts[0])
                    hght = float(parts[1])
                    temp = float(parts[2])
                    dwpt = float(parts[3])
                    data_rows.append({'P': pres, 'HGHT': hght, 'T': temp, 'TD': dwpt})
                except ValueError:
                    continue
                    
        if not data_rows:
            return None, "No valid data rows parsed."
            
        df = pd.DataFrame(data_rows)
        return df, None
        
    except Exception as e:
        return None, str(e)

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.title("⚙️ Parameters & Filters")

station_id = st.sidebar.text_input("Station WMO ID", value="40417")
date_input = st.sidebar.date_input("Date", pd.to_datetime("2026-08-22"))
time_option = st.sidebar.selectbox("UTC Time", ["00:00:00", "12:00:00"])

formatted_datetime = f"{date_input.strftime('%Y-%m-%d')}%20{time_option[:2]}:00:00"

st.sidebar.markdown("---")
st.sidebar.subheader("Duct Detection Thresholds")
min_delta_m = st.sidebar.slider("Min Delta M", min_value=1.0, max_value=20.0, value=5.0, step=0.5)
min_delta_z = st.sidebar.slider("Min Thickness / Delta Z [m]", min_value=10, max_value=200, value=40, step=5)

max_display_height = st.sidebar.number_input("Max Display Height [m]", min_value=500, max_value=10000, value=3000, step=250)

# ---------------------------------------------------------
# Main App Layout
# ---------------------------------------------------------
st.title("📡 Atmospheric Refractivity & Duct Analysis")
st.caption(f"Data Source: UWYO Station {station_id} | Selected Time: {date_input} {time_option}")

# Load Data
df, err = fetch_uwyo_sounding(station_id, formatted_datetime)

if err:
    st.error(f"Error fetching data: {err}")
else:
    # Process Refractivity
    df = calculate_refractivity(df)
    
    # Filter by height display limit
    df_filtered = df[df['HGHT'] <= max_display_height].reset_index(drop=True)
    
    # Run Detection Algorithm
    ducts_df = detect_ducts(
        df_filtered['HGHT'].values, 
        df_filtered['M'].values, 
        min_delta_m=min_delta_m, 
        min_delta_z=min_delta_z
    )
    
    # --- Top Summary Section ---
    st.subheader("📊 Layer Analysis Summary")
    
    if ducts_df is None or ducts_df.empty:
        st.info("🟢 **Standard Propagation Conditions** (No trapping layers detected with current criteria)")
    else:
        st.warning(f"⚠️ **{len(ducts_df)} Trapping Layer(s) Detected!**")
        st.dataframe(ducts_df, use_container_width=True)

    st.markdown("---")
    
    # --- Plotting Section ---
    st.subheader("📈 Refractivity Profiles")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Plot N Profile
        fig_n, ax_n = plt.subplots(figsize=(5, 6))
        ax_n.plot(df_filtered['N'], df_filtered['HGHT'], color='blue', linewidth=2, label='N (Refractivity)')
        ax_n.set_xlabel("Refractivity N [N-units]")
        ax_n.set_ylabel("Height AMSL [m]")
        ax_n.set_title("N Profile")
        ax_n.grid(True, linestyle='--', alpha=0.6)
        ax_n.legend(loc='upper right')
        st.pyplot(fig_n)
        
    with col2:
        # Plot M Profile
        fig_m, ax_m = plt.subplots(figsize=(5, 6))
        ax_m.plot(df_filtered['M'], df_filtered['HGHT'], color='red', linewidth=2, label='M (Modified Refractivity)')
        
        # Highlight Detected Ducts on Chart
        if ducts_df is not None and not ducts_df.empty:
            for _, row in ducts_df.iterrows():
                base_h = row['Base Alt [m]']
                top_h = row['Top Alt [m]']
                ax_m.axhspan(base_h, top_h, color='orange', alpha=0.3, label=f"{row['Duct Type']}")
        
        ax_m.set_xlabel("Modified Refractivity M [M-units]")
        ax_m.set_ylabel("Height AMSL [m]")
        ax_m.set_title("M Profile & Trapping Layers")
        ax_m.grid(True, linestyle='--', alpha=0.6)
        ax_m.legend(loc='upper right')
        st.pyplot(fig_m)

    # --- Raw Data Expander ---
    with st.expander("📄 View Parsed Sounding Data"):
        st.dataframe(df_filtered[['P', 'HGHT', 'T', 'TD', 'N', 'M']])
