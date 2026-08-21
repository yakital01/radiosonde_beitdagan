import math
import re
import urllib.parse
import urllib.request
from datetime import datetime, time
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# הגדרות תצוגה מותאמות למובייל
st.set_page_config(
    page_title="פרופיל M ו-N - נתוני רדיוסונדה עולמיים", layout="wide"
)


# --- פונקציות חישוב פיזיקליות ---
def calculate_N(p, T_C, RH):
    """מחשב את מקדם השבירה האטמוספרי N"""
    T_K = T_C + 273.15
    # לחץ אדים רווי (hPa)
    e_sat = 6.112 * math.exp((17.67 * T_C) / (T_C + 243.5))
    e = (RH / 100.0) * e_sat
    # נוסחת N המקובלת
    N = (77.6 * p / T_K) + (3.73e5 * e / (T_K**2))
    return N


def calculate_M(N, z_m):
    """מחשב את מקדם השבירה המותאם M"""
    return N + 0.157 * z_m


# --- פונקציית שליפת נתונים מ-UWYO ---
def fetch_uwyo_data(station_id, date_obj, hour_str, src_type):
    year = date_obj.strftime("%Y")
    month = date_obj.strftime("%m")
    day_from = date_obj.strftime("%d")
    day_to = date_obj.strftime("%d")

    if hour_str == "00":
        from_hour, to_hour = "00", "00"
    elif hour_str == "12":
        from_hour, to_hour = "12", "12"
    elif hour_str == "06":
        from_hour, to_hour = "06", "06"
    elif hour_str == "18":
        from_hour, to_hour = "18", "18"
    else:
        from_hour, to_hour = "00", "12"

    params = {
        "region": "namer",  # ברירת מחדל לאזור - האתר מטפל בחיפוש לפי station_id
        "TYPE": "TEXT:LIST",
        "YEAR": year,
        "MONTH": month,
        "FROM": f"{day_from}{from_hour}",
        "TO": f"{day_to}{to_hour}",
        "STNM": station_id,
        "src": src_type,
    }

    base_url = (
        "http://weather.uwyo.edu/cgi-bin/sounding"  # HTTP יציב עבור פקודת fetch
    )
    full_url = f"{base_url}?{urllib.parse.urlencode(params)}"

    req = urllib.request.Request(
        full_url, headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req) as response:
        html = response.read().decode("utf-8", errors="ignore")

    soup = BeautifulSoup(html, "html.parser")
    pre_tag = soup.find("pre")

    if not pre_tag:
        return None, full_url

    text_data = pre_tag.get_text()
    lines = text_data.strip().split("\n")

    # חיפוש שורת הטיטרציה/כותרות הטבלה
    data_rows = []
    start_parsing = False
    for line in lines:
        if "PRES" in line and "HGHT" in line:
            start_parsing = True
            continue
        if start_parsing:
            if line.startswith("------") or line.startswith("SHOW"):
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    p = float(parts[0])
                    z = float(parts[1])
                    t = float(parts[2])
                    rh = float(parts[4])
                    data_rows.append(
                        {"PRES": p, "HGHT": z, "TEMP": t, "RELH": rh}
                    )
                except ValueError:
                    continue

    if not data_rows:
        return None, full_url

    df = pd.DataFrame(data_rows)
    return df, full_url


# --- פונקציית דילול מדורג לפי גובה ---
def filter_high_res_data(df):
    """
    מבצע דילול מדורג אך ורק במידה והנתונים ברזולוציה גבוהה (מתחת ל-25 מטר הפרש ממוצע)
    """
    df = df.sort_values("HGHT").reset_index(drop=True)

    # בדיקת הפרשי גובה ממוצעים ב-10 השורות הראשונות
    if len(df) > 10:
        diffs = df["HGHT"].diff().dropna()
        avg_diff = diffs.head(10).mean()
    else:
        avg_diff = 100  # אם מעט מדי שורות, אל תדלל

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
            # מעל 3,000 מטר לוקחים את כל הנתונים הקיימים
            filtered_rows.append(row)
            last_z = z

    return pd.DataFrame(filtered_rows)


# --- ממשק משתמש ב-Streamlit ---
st.title("📊 פרופיל N ו-M מנתוני רדיוסונדה (UWYO)")

# סרגל צד להזנת נתונים - מותאם למובייל
st.sidebar.header("הגדרות שליפה")

# בחירת תחנה מקובלת או הקלדה חופשית
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
        # קישור ישיר לדף UWYO לנוחות
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

        # הצגת הגרפים בטאבים
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

            # שנתות מדויקות
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
