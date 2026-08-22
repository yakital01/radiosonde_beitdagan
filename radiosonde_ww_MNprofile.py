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


def calculate_cutoff_frequency(delta_z, delta_m):
    """
    מחשב את תדר הקטעון (Cut-off Frequency) במגה-הרץ (MHz)
    f_c = c / (0.157 * (delta_z^1.5) * sqrt(delta_m))
    """
    if delta_z <= 0 or delta_m <= 0:
        return None

    c = 3e8  # מהירות האור במטרים לשניה
    try:
        f_hz = c / (0.157 * (delta_z**1.5) * math.sqrt(delta_m))
        f_mhz = f_hz / 1e6
        return f_mhz
    except ZeroDivisionError:
        return None


def calculate_critical_angle(delta_m):
    """
    מחשב את הזווית הקריטית במעלות (Critical Angle in Degrees)
    theta_c = 0.081 * sqrt(delta_m)
    """
    if delta_m <= 0:
        return 0.0
    return 0.081 * math.sqrt(delta_m)


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


# --- אלגוריתם ניתוח וזיהוי תעלות ---
def detect_ducts(df_input, min_delta_m=5.0, min_delta_z=50.0):
    if df_input is None or len(df_input) < 2:
        return []

    df = df_input.sort_values("HGHT").reset_index(drop=True).copy()

    surface_z = df.loc[0, "HGHT"]
    surface_m = df.loc[0, "M"]

    raw_ducts = []
    in_duct = False
    start_idx = 0

    # 1. זיהוי מקטעי שיפוע שלילי (dM/dz < 0) בנתונים المקוריים
    for i in range(len(df) - 1):
        dM = df.loc[i + 1, "M"] - df.loc[i, "M"]

        if dM < 0:
            if not in_duct:
                in_duct = True
                start_idx = i
        else:
            if in_duct:
                in_duct = False
                end_idx = i
                raw_ducts.append((start_idx, end_idx))

    if in_duct:
        raw_ducts.append((start_idx, len(df) - 1))

    # 2. איחוד תעלות סמוכות מאוד (מרווח מופרד קטן מ-30 מטר)
    merged_ducts = []
    for duct in raw_ducts:
        if not merged_ducts:
            merged_ducts.append(duct)
        else:
            prev_start, prev_end = merged_ducts[-1]
            curr_start, curr_end = duct
            gap_z = df.loc[curr_start, "HGHT"] - df.loc[prev_end, "HGHT"]
            if gap_z <= 30.0:
                merged_ducts[-1] = (prev_start, curr_end)
            else:
                merged_ducts.append(duct)

    # 3. סינון לפי ספי מינימום וסיווג פיזיקלי
    final_ducts = []

    for s_idx, e_idx in merged_ducts:
        z_base = df.loc[s_idx, "HGHT"]
        z_top = df.loc[e_idx, "HGHT"]
        m_base = df.loc[s_idx, "M"]
        m_top = df.loc[e_idx, "M"]

        delta_z = z_top - z_base
        delta_m = m_base - m_top

        # סינון לפי עובי ועוצמה מינימליים
        if delta_z < min_delta_z or delta_m < min_delta_m:
            continue

        # סיווג סוג התעלה (באנגלית)
        if s_idx == 0 or abs(z_base - surface_z) < 1.0:
            duct_type = "Surface Duct"
        elif m_top <= surface_m:
            duct_type = "Surface-Based Duct"
        else:
            duct_type = "Elevated Duct"

        # חישוב תדר קטעון וזווית קריטית
        fc_mhz = calculate_cutoff_frequency(delta_z, delta_m)
        crit_angle_deg = calculate_critical_angle(delta_m)

        final_ducts.append(
            {
                "type": duct_type,
                "z_base": z_base,
                "z_top": z_top,
                "delta_z": delta_z,
                "m_base": m_base,
                "m_top": m_top,
                "delta_m": round(delta_m),  # עיגול למספר שלם
                "f_cutoff_mhz": fc_mhz,
                "crit_angle_deg": crit_angle_deg,
            }
        )

    return final_ducts


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

# --- הגדרות סינון תעלות ב-Sidebar ---
st.sidebar.markdown("---")
st.sidebar.header("⚙️ ספי זיהוי תעלות (Ducting)")
min_delta_m = st.sidebar.slider(
    "עוצמה מינימלית (ΔM min):", 0.5, 15.0, 5.0, step=0.5
)
min_delta_z = st.sidebar.slider(
    "עובי מינימלי (ΔZ min במטרים):", 10, 200, 50, step=10
)

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

        formatted_date_str = selected_date.strftime("%d%b%Y")
        datetime_str = f"{formatted_date_str} {selected_hour}:00Z"

        # הרצת אלגוריתם זיהוי התעלות
        detected_ducts = detect_ducts(
            df_proc, min_delta_m=min_delta_m, min_delta_z=min_delta_z
        )
        display_ducts = [d for d in detected_ducts if d["z_base"] <= max_height]

        tab1, tab2, tab3 = st.tabs(
            [
                "📈 פרופיל M וזיהוי תעלות",
                "📉 פרופיל N",
                "📋 טבלת נתונים מעובדת",
            ]
        )

        csv_data = df_plot[["HGHT", "PRES", "TEMP", "RELH", "N", "M"]].to_csv(
            index=False
        )

        with tab1:
            col_graph, col_info = st.columns([1.6, 1])

            with col_graph:
                fig, ax = plt.subplots(figsize=(6, 8))
                ax.plot(
                    df_plot["M"],
                    df_plot["HGHT"],
                    color="blue",
                    linewidth=1.8,
                    marker="o",
                    markersize=3,
                    label="M Profile",
                )

                colors_map = {
                    "Surface Duct": "red",
                    "Surface-Based Duct": "orange",
                    "Elevated Duct": "purple",
                }

                for d in display_ducts:
                    c = colors_map.get(d["type"], "red")
                    ax.axhspan(
                        d["z_base"],
                        min(d["z_top"], max_height),
                        color=c,
                        alpha=0.2,
                        label=f"{d['type']} ({d['z_base']:.0f}-{d['z_top']:.0f}m)",
                    )
                    ax.axhline(
                        d["z_base"],
                        color=c,
                        linestyle="--",
                        linewidth=1.2,
                        alpha=0.7,
                    )
                    ax.axhline(
                        min(d["z_top"], max_height),
                        color=c,
                        linestyle="--",
                        linewidth=1.2,
                        alpha=0.7,
                    )

                ax.set_xlabel("M (Modified Refractivity)")
                ax.set_ylabel("Height (m)")
                ax.set_title(
                    f"Modified Refractivity (M) Profile\nStation: {station_id}"
                    f" | Date: {datetime_str}"
                )
                ax.grid(True, which="both", linestyle="--", alpha=0.6)
                ax.legend(loc="upper right", fontsize="small")

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
                        label="🖼️ הורד גרף M (PNG)",
                        data=buf_m,
                        file_name=f"M_profile_{station_id}_{formatted_date_str}_{selected_hour}Z.png",
                        mime="image/png",
                        use_container_width=True,
                    )
                with col2:
                    st.download_button(
                        label="📄 הורד נתוני CSV",
                        data=csv_data,
                        file_name=f"M_profile_{station_id}_{formatted_date_str}_{selected_hour}Z.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

            with col_info:
                st.subheader("📡 Ducting Analysis")

                if not detected_ducts:
                    st.info("Standard Propagation Conditions")
                else:
                    st.success(
                        f"Detected **{len(detected_ducts)}** Trapping Layer(s)!"
                    )

                    duct_table_data = []

                    for idx, d in enumerate(detected_ducts, 1):
                        fc_str = (
                            f"{d['f_cutoff_mhz']:.1f} MHz"
                            if d["f_cutoff_mhz"] and d["f_cutoff_mhz"] < 1000
                            else f"{d['f_cutoff_mhz']/1000:.2f} GHz"
                            if d["f_cutoff_mhz"]
                            else "N/A"
                        )
                        crit_angle_str = f"{d['crit_angle_deg']:.2f}°"

                        with st.expander(
                            f"Layer {idx}: {d['type']} ({d['z_base']:.0f}m -"
                            f" {d['z_top']:.0f}m)",
                            expanded=True,
                        ):
                            m1, m2 = st.columns(2)
                            m1.metric("Thickness (ΔZ)", f"{d['delta_z']:.0f} m")
                            m2.metric("Intensity (ΔM)", f"{d['delta_m']}")

                            m3, m4 = st.columns(2)
                            m3.metric("Base Alt (z_base)", f"{d['z_base']:.0f} m")
                            m4.metric("Top Alt (z_top)", f"{d['z_top']:.0f} m")

                            st.markdown(
                                f"**Cutoff Frequency ($f_{{cutoff}}$):** `{fc_str}`"
                            )
                            st.markdown(
                                f"**Critical Angle ($\theta_c$):** `{crit_angle_str}`"
                            )

                        duct_table_data.append(
                            {
                                "#": idx,
                                "Duct Type": d["type"],
                                "Base Alt [m]": f"{d['z_base']:.0f}",
                                "Top Alt [m]": f"{d['z_top']:.0f}",
                                "Thickness ΔZ [m]": f"{d['delta_z']:.0f}",
                                "Delta M": f"{d['delta_m']}",
                                "Cutoff Freq": fc_str,
                                "Critical Angle": crit_angle_str,
                            }
                        )

                    st.markdown("#### 📊 Summary Table")
                    st.dataframe(
                        pd.DataFrame(duct_table_data), hide_index=True
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
                    label="🖼️ הורד גרף N (PNG)",
                    data=buf_n,
                    file_name=f"N_profile_{station_id}_{formatted_date_str}_{selected_hour}Z.png",
                    mime="image/png",
                    use_container_width=True,
                )
            with col2_n:
                st.download_button(
                    label="📄 הורד נתוני CSV",
                    data=csv_data,
                    file_name=f"N_profile_{station_id}_{formatted_date_str}_{selected_hour}Z.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        with tab3:
            st.dataframe(df_plot[["HGHT", "PRES", "TEMP", "RELH", "N", "M"]])
