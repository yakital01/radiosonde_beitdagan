import math
import re
import urllib.parse
from datetime import datetime
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

# הגדרות תצוגה מותאמות למובייל
st.set_page_config(
    page_title="פרופיל M ו-N - נתוני רדיוסונדה עולמיים", layout="wide"
)


# --- פונקציות חישוב פיזיקליות ---
def calculate_N(p, T_C, RH):
    """מחשב את מקדם השבירה האטמוספרי N"""
    T_K = T_C + 273.15
    e_sat = 6.112 * math.exp((17.67 * T_C) / (T_C + 243.5))
    e = (RH / 100.0) * e_sat
    N = (77.6 * p / T_K) + (3.73e5 * e / (T_K**2))
    return N


def calculate_M(N, z_m):
    """מחשב את מקדם השבירה המותאם M"""
    return N + 0.157 * z_m


# --- פונקציית שליפת נתונים מ-UWYO ---
def fetch_uwyo_data(station_id, date_obj, hour_str, preferred_src="BUFR"):
    date_str = date_obj.strftime("%Y-%m-%d")
    raw_datetime = f"{date_str} {hour_str}:00:00"
    encoded_datetime = urllib.parse.quote(raw_datetime)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            " (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    # מנגנון ניסיון מועדף + נפילה (Fallback) למקור השני אם נכשל
    sources_to_try = [preferred_src]
    alt_src = "FM35" if preferred_src == "BUFR" else "BUFR"
    sources_to_try.append(alt_src)

    for src_type in sources_to_try:
        full_url = f"https://weather.uwyo.edu/wsgi/sounding?datetime={encoded_datetime}&id={station_id}&src={src_type}&type=TEXT:LIST"

        try:
            response = requests.get(full_url, headers=headers, timeout=15)
            if response.status_code != 200:
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            pre_tag = soup.find("pre")

            if not pre_tag or "Unable to retrieve" in response.text:
                continue

            lines = pre_tag.text.strip().split("\n")
            data_rows = []

            for line in lines:
                parts = line.split()
                # וידוא שורת נתונים: לפחות 5 עמודות, והראשונה היא מספר (לחץ)
                if len(parts) >= 5:
                    try:
                        p = float(parts[0])  # PRES
                        z = float(parts[1])  # HGHT
                        t = float(parts[2])  # TEMP
                        rh = float(parts[4])  # RELH

                        # סינון ערכים לא תקינים או חסרים (כמו 9999)
                        if p < 100 or z < -100 or t < -100 or rh < 0:
                            continue

                        data_rows.append(
                            {"PRES": p, "HGHT": z, "TEMP": t, "RELH": rh}
                        )
                    except ValueError:
                        continue

            if data_rows:
                df = pd.DataFrame(data_rows)
                return df, full_url

        except Exception as e:
            print(f"Error fetching {src_type}: {e}")
            continue

    # אם שני המקורות נכשלו
    last_url = f"https://weather.uwyo.edu/wsgi/sounding?datetime={encoded_datetime}&id={station_id}&src={preferred_src}&type=TEXT:LIST"
    return None, last_url


# --- פונקציית דילול מדורג לפי גובה ---
def filter_high_res_data(df):
    """
    מבצע דילול מדורג אך ורק במידה והנתונים ברזולוציה גבוהה (מתחת ל-25 מטר הפרש ממוצע)
    """
    df = df.sort_values("HGHT").reset_index(drop=True)

    if len(df) > 10:
        diffs = df["HGHT"].diff().dropna()
        avg_diff = diffs.head(10).mean()
    else:
        avg_diff = 100

    # אם הרזולוציה גסה מ-25 מטר - החזר ללא דילול
    if avg_diff > 25:
        return df

    filtered_rows = []
    last_z = -9999

    for idx, row in df.iterrows():
        z = row["HGHT"]
        if last_z == -9999:
            filtered_rows.append(row)
            last_z = z
            continue

        step = 0
        if z <= 1000:
            step = 25
        elif z <= 2000:
            step = 50
        elif z <= 3000:
            step = 100

        if step > 0:
            if (z - last_z) >= step:
                filtered_rows.append(row)
                last_z = z
        else:
            filtered_rows.append(row)
            last_z = z

    return pd.DataFrame(filtered_rows)


# --- ממשק משתמש ב-Streamlit ---
st.title("📊 פרופיל N ו-M מנתוני רדיוסונדה (UWYO)")

st.sidebar.header("הגדרות שליפה")

stations = {
    "40179 - בית דגן (ישראל)": "40179",
    "10393 - לנדסברג (גרמניה)": "10393",
    "16080 - מילאנו (איטליה)": "16080",
    "62318 - אל עריש (מצרים)": "62318",
    "הקלדה ידנית...": "custom",
}

station_choice = st.sidebar.selectbox("בחר תחנה:", list(stations.keys()))

if stations[station_choice] == "custom":
    station_id = st.sidebar.text_input("הכנס קוד תחנה (WMO ID):", value="40179")
else:
    station_id = stations[station_choice]

selected_date = st.sidebar.date_input("תאריך:", datetime.today())
selected_hour = st.sidebar.selectbox(
    "שעה (UTC):", ["12", "00", "06", "18"], index=0
)
src_type = st.sidebar.radio(
    "מקור נתונים (src):",
    ["BUFR (רזולוציה גבוהה)", "FM35 (רזולוציה סטנדרטית)"],
    index=0,
)
src_code = "BUFR" if "BUFR" in src_type else "FM35"

submit_btn = st.sidebar.button("🚀 שליפה והצגת פרופיל")

if submit_btn:
    with st.spinner("שולף נתונים מאתר UWYO..."):
        df, uwyo_url = fetch_uwyo_data(
            station_id, selected_date, selected_hour, src_code
        )

    if df is None or df.empty:
        st.error(
            "לא נמצאו נתונים לתחנה ולזמן הנבחר. ייתכן והנתונים עדיין לא עודכנו באתר."
        )
        st.markdown(f"🔗 [לבדיקת הדף המקורי באתר UWYO]({uwyo_url})")
    else:
        st.success("הנתונים נשלפו בהצלחה!")
        st.markdown(
            f"🔗 **[לחץ כאן לצפייה בטבלת הנתונים המקורית באתר UWYO]({uwyo_url})**"
        )

        # דילול מדורג
        df_filtered = filter_high_res_data(df)

        # חישוב N ו-M
        df_filtered["N"] = df_filtered.apply(
            lambda r: calculate_N(r["PRES"], r["TEMP"], r["RELH"]), axis=1
        )
        df_filtered["M"] = df_filtered.apply(
            lambda r: calculate_M(r["N"], r["HGHT"]), axis=1
        )

        tab1, tab2, tab3 = st.tabs(
            ["📈 פרופיל M", "📉 פרופיל N", "📋 טבלת נתונים מעובדת"]
        )

        with tab1:
            fig, ax = plt.subplots(figsize=(6, 8))
            ax.plot(
                df_filtered["M"],
                df_filtered["HGHT"],
                color="blue",
                linewidth=1.8,
                marker="o",
                markersize=3,
            )
            ax.set_xlabel("M (Modified Refractivity)")
            ax.set_ylabel("Height (m)")
            ax.set_title(f"Modified Refractivity (M) Profile - {station_id}")
            ax.grid(True, which="both", linestyle="--", alpha=0.6)

            m_min, m_max = math.floor(
                df_filtered["M"].min() / 50
            ) * 50, math.ceil(df_filtered["M"].max() / 50) * 50
            if m_max - m_min >= 50:
                ax.set_xticks(np.arange(m_min, m_max + 1, 50))

            st.pyplot(fig)

        with tab2:
            fig_n, ax_n = plt.subplots(figsize=(6, 8))
            ax_n.plot(
                df_filtered["N"],
                df_filtered["HGHT"],
                color="green",
                linewidth=1.8,
                marker="o",
                markersize=3,
            )
            ax_n.set_xlabel("N (Refractivity)")
            ax_n.set_ylabel("Height (m)")
            ax_n.set_title(f"Refractivity (N) Profile - {station_id}")
            ax_n.grid(True, which="both", linestyle="--", alpha=0.6)

            st.pyplot(fig_n)

        with tab3:
            st.dataframe(df_filtered[["HGHT", "PRES", "TEMP", "RELH", "N", "M"]])
