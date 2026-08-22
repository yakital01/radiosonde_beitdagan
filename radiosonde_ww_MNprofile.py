import calendar
import io
import math
import re
import time
import urllib.parse
from datetime import datetime
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

# הגדרות תצוגה
st.set_page_config(
    page_title="World Wide Radiosonde Refractivity Profiles (N & M)",
    layout="wide",
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

    sources_to_try = [preferred_src]
    alt_src = "FM35" if preferred_src == "BUFR" else "BUFR"
    sources_to_try.append(alt_src)

    for src_type in sources_to_try:
        full_url = f"https://weather.uwyo.edu/wsgi/sounding?datetime={encoded_datetime}&id={station_id}&src={src_type}&type=TEXT:LIST"

        for attempt in range(3):
            try:
                response = requests.get(full_url, headers=headers, timeout=45)
                if response.status_code != 200:
                    time.sleep(1.5)
                    continue

                soup = BeautifulSoup(response.text, "html.parser")
                pre_tag = soup.find("pre")

                if not pre_tag or "Unable to retrieve" in response.text:
                    break

                lines = pre_tag.text.strip().split("\n")
                data_rows = []

                for line in lines:
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            p = float(parts[0])
                            z = float(parts[1])
                            t = float(parts[2])
                            rh = float(parts[4])

                            if p < 100 or z < -100 or t < -100 or rh < 0:
                                continue

                            data_rows.append(
                                {"PRES": p, "HGHT": z, "TEMP": t, "RELH": rh}
                            )
                        except ValueError:
                            continue

                if data_rows:
                    return pd.DataFrame(data_rows), full_url

            except Exception:
                time.sleep(1.5)

    last_url = f"https://weather.uwyo.edu/wsgi/sounding?datetime={encoded_datetime}&id={station_id}&src={preferred_src}&type=TEXT:LIST"
    return None, last_url


# --- פונקציית דילול מדורג לפי גובה ---
def filter_high_res_data(df):
    df = df.sort_values("HGHT").reset_index(drop=True)

    if len(df) > 10:
        diffs = df["HGHT"].diff().dropna()
        avg_diff = diffs.head(10).mean()
    else:
        avg_diff = 100

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


# --- פונקציה לחיתוך מדויק ואינטרפולציה בגובה היעד ---
def crop_and_interpolate(df, max_hght):
    df = df.sort_values("HGHT").reset_index(drop=True)

    df_below = df[df["HGHT"] <= max_hght].copy()
    df_above = df[df["HGHT"] > max_hght].copy()

    if not df_below.empty and not df_above.empty:
        p1 = df_below.iloc[-1]
        p2 = df_above.iloc[0]

        if p1["HGHT"] < max_hght:
            z1, z2 = p1["HGHT"], p2["HGHT"]
            factor = (max_hght - z1) / (z2 - z1)

            interp_row = {
                "HGHT": max_hght,
                "PRES": p1["PRES"] + factor * (p2["PRES"] - p1["PRES"]),
                "TEMP": p1["TEMP"] + factor * (p2["TEMP"] - p1["TEMP"]),
                "RELH": p1["RELH"] + factor * (p2["RELH"] - p1["RELH"]),
                "N": p1["N"] + factor * (p2["N"] - p1["N"]),
                "M": p1["M"] + factor * (p2["M"] - p1["M"]),
            }
            df_below = pd.concat(
                [df_below, pd.DataFrame([interp_row])], ignore_index=True
            )

    return df_below


# --- ממשק משתמש ב-Streamlit ---
st.title("🌐 World Wide Radiosonde Refractivity Profiles (N & M)")

st.sidebar.header("הגדרות שליפה")

stations = {
    "40179 - בית דגן (ישראל)": "40179",
    "40417 - עמאן (ירדן)": "40417",
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

# --- מנגנון בחירת תאריך מוגן מקלדת ---
st.sidebar.markdown("**תאריך:**")
today = datetime.today()

col_y, col_m, col_d = st.sidebar.columns(3)

with col_y:
    selected_year = st.selectbox(
        "שנה", list(range(today.year, today.year - 10, -1)), index=0
    )

with col_m:
    selected_month = st.selectbox(
        "חודש", list(range(1, 13)), index=today.month - 1
    )

max_days_in_month = calendar.monthrange(selected_year, selected_month)[1]

with col_d:
    default_day_idx = min(today.day, max_days_in_month) - 1
    selected_day = st.selectbox(
        "יום", list(range(1, max_days_in_month + 1)), index=default_day_idx
    )

selected_date = datetime(
    selected_year, selected_month, selected_day
).date()

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
        raw_df, uwyo_url = fetch_uwyo_data(
            station_id, selected_date, selected_hour, src_code
        )

        if raw_df is not None and not raw_df.empty:
            df_proc = filter_high_res_data(raw_df)
            df_proc["N"] = df_proc.apply(
                lambda r: calculate_N(r["PRES"], r["TEMP"], r["RELH"]), axis=1
            )
            df_proc["M"] = df_proc.apply(
                lambda r: calculate_M(r["N"], r["HGHT"]), axis=1
            )

            st.session_state["processed_df"] = df_proc
            st.session_state["uwyo_url"] = uwyo_url
        else:
            st.session_state["processed_df"] = None
            st.session_state["uwyo_url"] = uwyo_url

if "processed_df" in st.session_state:
    df_proc = st.session_state["processed_df"]
    uwyo_url = st.session_state["uwyo_url"]

    if df_proc is None or df_proc.empty:
        st.error(
            "לא נמצאו נתונים לתחנה ולזמן הנבחר. ייתכן והנתונים עדיין לא עודכנו באתר או שהשרת חווה עומס."
        )
        st.markdown(f"🔗 [לבדיקת הדף המקורי באתר UWYO]({uwyo_url})")
    else:
        st.success("הנתונים נשלפו בהצלחה!")
        st.markdown(
            f"🔗 **[לחץ כאן לצפייה בטבלת הנתונים המקורית באתר UWYO]({uwyo_url})**"
        )

        st.markdown("---")

        max_height = st.radio(
            "🔍 בחירת זום - גובה מקסימלי לתצוגה (מטרים):",
            options=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000],
            index=4,
            horizontal=True,
        )

        df_plot = crop_and_interpolate(df_proc, max_height)

        # עיצוב התאריך לפי הפורמט המבוקש: DDMonYYYY (למשל 22Aug2026)
        formatted_date_str = selected_date.strftime("%d%b%Y")
        datetime_str = f"{formatted_date_str} {selected_hour}:00Z"

        tab1, tab2, tab3 = st.tabs(
            ["📈 פרופיל M", "📉 פרופיל N", "📋 טבלת נתונים מעובדת"]
        )

        # המרת הנתונים ל-CSV
        csv_data = df_plot[["HGHT", "PRES", "TEMP", "RELH", "N", "M"]].to_csv(
            index=False
        )

        with tab1:
            fig, ax = plt.subplots(figsize=(6, 8))
            ax.plot(
                df_plot["M"],
                df_plot["HGHT"],
                color="blue",
                linewidth=1.8,
                marker="o",
                markersize=3,
            )
            ax.set_xlabel("M (Modified Refractivity)")
            ax.set_ylabel("Height (m)")
            ax.set_title(
                f"Modified Refractivity (M) Profile\nStation: {station_id} |"
                f" Date: {datetime_str}"
            )
            ax.grid(True, which="both", linestyle="--", alpha=0.6)

            if not df_plot.empty:
                m_min = math.floor(df_plot["M"].min() / 50) * 50
                m_max = math.ceil(df_plot["M"].max() / 50) * 50
                if m_max - m_min >= 50:
                    ax.set_xticks(np.arange(m_min, m_max + 1, 50))
                ax.set_ylim(df_plot["HGHT"].min(), max_height)

            st.pyplot(fig)

            buf_m = io.BytesIO()
            fig.savefig(buf_m, format="png", dpi=300, bbox_inches="tight")
            buf_m.seek(0)

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="🖼️ הורד גרף פרופיל M (PNG)",
                    data=buf_m,
                    file_name=f"M_profile_{station_id}_{formatted_date_str}_{selected_hour}Z.png",
                    mime="image/png",
                    use_container_width=True,
                )
            with col2:
                st.download_button(
                    label="📄 הורד נתוני גרף M (CSV)",
                    data=csv_data,
                    file_name=f"M_profile_{station_id}_{formatted_date_str}_{selected_hour}Z.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        with tab2:
            fig_n, ax_n = plt.subplots(figsize=(6, 8))
            ax_n.plot(
                df_plot["N"],
                df_plot["HGHT"],
                color="green",
                linewidth=1.8,
                marker="o",
                markersize=3,
            )
            ax_n.set_xlabel("N (Refractivity)")
            ax_n.set_ylabel("Height (m)")
            ax_n.set_title(
                f"Refractivity (N) Profile\nStation: {station_id} | Date:"
                f" {datetime_str}"
            )
            ax_n.grid(True, which="both", linestyle="--", alpha=0.6)

            if not df_plot.empty:
                n_min = math.floor(df_plot["N"].min() / 50) * 50
                n_max = math.ceil(df_plot["N"].max() / 50) * 50
                if n_max - n_min >= 50:
                    ax_n.set_xticks(np.arange(n_min, n_max + 1, 50))
                ax_n.set_ylim(df_plot["HGHT"].min(), max_height)

            st.pyplot(fig_n)

            buf_n = io.BytesIO()
            fig_n.savefig(buf_n, format="png", dpi=300, bbox_inches="tight")
            buf_n.seek(0)

            col1_n, col2_n = st.columns(2)
            with col1_n:
                st.download_button(
                    label="🖼️ הורד גרף פרופיל N (PNG)",
                    data=buf_n,
                    file_name=f"N_profile_{station_id}_{formatted_date_str}_{selected_hour}Z.png",
                    mime="image/png",
                    use_container_width=True,
                )
            with col2_n:
                st.download_button(
                    label="📄 הורד נתוני גרף N (CSV)",
                    data=csv_data,
                    file_name=f"N_profile_{station_id}_{formatted_date_str}_{selected_hour}Z.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        with tab3:
            st.dataframe(df_plot[["HGHT", "PRES", "TEMP", "RELH", "N", "M"]])
