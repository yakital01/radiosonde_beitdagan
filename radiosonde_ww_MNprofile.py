import numpy as np
import pandas as pd

def detect_ducts(z_arr, m_arr, min_delta_m=5.0, min_delta_z=40.0, c_speed=3e8):
    """
    Detects Atmospheric Ducts (Surface & Elevated) with English Output.
    
    Parameters:
      z_arr : array-like, Heights in meters [m]
      m_arr : array-like, Modified Refractivity [M-units]
      min_delta_m : float, Threshold for Delta M
      min_delta_z : float, Threshold for Delta Z [m]
      c_speed : float, Speed of light (3e8 m/s)
      
    Returns:
      pandas.DataFrame containing detected ducts or string if none found.
    """
    z = np.array(z_arr, dtype=float)
    m = np.array(m_arr, dtype=float)
    
    ducts = []
    n = len(z)
    
    i = 0
    while i < n - 1:
        # Check if M decreases in the next step (Negative Gradient)
        if m[i+1] < m[i]:
            base_idx = i
            base_z = z[base_idx]
            base_m = m[base_idx]
            
            # Find the minimum M point (Trapping Layer Top)
            j = i + 1
            while j < n - 1 and m[j+1] < m[j]:
                j += 1
            
            top_idx = j
            top_z = z[top_idx]
            top_m = m[top_idx]
            
            delta_m = base_m - top_m
            layer_thickness = top_z - base_z
            
            # Apply filter thresholds
            if delta_m >= min_delta_m and layer_thickness >= min_delta_z:
                
                # Determine Duct Type
                # If base is at the ground level (index 0), it's a Surface Duct
                if base_idx == 0:
                    duct_type = "Surface Duct"
                    d_eff = layer_thickness  # Effective thickness
                else:
                    # Search for reference height where M(z) == M_top to find bottom edge of elevated duct
                    # If M drops below base level, calculate effective duct thickness
                    duct_type = "Elevated Duct"
                    d_eff = layer_thickness
                
                # Cutoff Frequency Calculation (Hz and GHz)
                # Formula: fc = (2/3) * c / (d_eff * sqrt(2 * 1e-6 * delta_m))
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
            
            # Move index forward
            i = top_idx
        else:
            i += 1

    if not ducts:
        return "Standard Propagation Conditions"
    
    return pd.DataFrame(ducts)

# --- בדיקה על נתוני הדוגמה שלך (בית דגן 22/08/2026 00Z) ---
z_sample = [12, 113, 230, 450, 800]
m_sample = [351, 311, 325, 340, 360]

# הרצת הפונקציה
results = detect_ducts(z_sample, m_sample, min_delta_m=5.0, min_delta_z=40.0)

if isinstance(results, str):
    print(f"Status: {results}")
else:
    print("Duct Detection Results:")
    print(results.to_string(index=False))
