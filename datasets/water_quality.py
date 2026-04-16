# =====================================================
# Balanced Water Quality Dataset Generator (1500 samples)
# =====================================================

import pandas as pd
import numpy as np

np.random.seed(42)


# =====================================================
# CATEGORY FUNCTION
# =====================================================

def get_category(value, param):

    if param == "pH":
        if 7.1 <= value <= 7.8:
            return "Good"
        elif (6.5 <= value <= 7.0) or (7.9 <= value <= 8.5):
            return "Moderate"
        else:
            return "Poor"

    elif param == "TDS":
        if 200 <= value <= 270:
            return "Good"
        elif (150 <= value <= 199) or (271 <= value <= 350):
            return "Moderate"
        else:
            return "Poor"

    elif param == "Turbidity":
        if value < 1:
            return "Good"
        elif 1 <= value < 5:
            return "Moderate"
        else:
            return "Poor"

    elif param == "MP":
        if value == 0:
            return "Good"
        elif 1 <= value <= 5:
            return "Moderate"
        else:
            return "Poor"


# =====================================================
# RISK FUNCTION
# =====================================================

def assign_risk(ph_c, tds_c, turb_c, mp_c):

    categories = [ph_c, tds_c, turb_c, mp_c]

    if "Poor" in categories:
        return "High"
    elif categories.count("Moderate") >= 2:
        return "Medium"
    else:
        return "Low"


# =====================================================
# CONTROLLED GENERATION (BALANCED DATASET)
# =====================================================

def generate_sample(target_risk):

    while True:

        # Generate raw values
        pH = round(np.random.uniform(4.5, 10.0), 1)
        TDS = round(np.random.uniform(50, 800), 1)
        Turbidity = round(np.random.uniform(0, 15), 1)
        MP = np.random.randint(0, 30)

        # Get categories
        ph_c = get_category(pH, "pH")
        tds_c = get_category(TDS, "TDS")
        turb_c = get_category(Turbidity, "Turbidity")
        mp_c = get_category(MP, "MP")

        risk = assign_risk(ph_c, tds_c, turb_c, mp_c)

        # Keep only required risk type
        if risk == target_risk:
            return {
                "pH": pH,
                "TDS": TDS,
                "Turbidity": Turbidity,
                "MP_Count": MP,
                "pH_cat": ph_c,
                "TDS_cat": tds_c,
                "Turbidity_cat": turb_c,
                "MP_cat": mp_c,
                "Risk": risk
            }


# =====================================================
# DATASET GENERATION (BALANCED)
# =====================================================

data = []

print("Generating balanced dataset...")

for _ in range(500):
    data.append(generate_sample("Low"))
    data.append(generate_sample("Medium"))
    data.append(generate_sample("High"))

df = pd.DataFrame(data)

# Shuffle dataset
df = df.sample(frac=1).reset_index(drop=True)

# =====================================================
# SAVE DATASET
# =====================================================

df.to_csv("water_quality_dataset.csv", index=False)

print("\nDataset saved successfully ✔")
print("\nSample:")
print(df.head())