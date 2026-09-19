# =====================================================
# Balanced Water Quality Dataset Generator (1500 samples)
# =====================================================

import pandas as pd
import numpy as np

np.random.seed(42)


# =====================================================
# 1. CATEGORY FUNCTION
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
# 2. RISK FUNCTION
# =====================================================

def assign_risk(ph_c, tds_c, turb_c, mp_c):

    categories = [ph_c, tds_c, turb_c, mp_c]

    # Any Poor parameter = High Risk
    if "Poor" in categories:
        return "High"

    # Two or more Moderate parameters = Medium Risk
    elif categories.count("Moderate") >= 2:
        return "Medium"

    # Otherwise Low Risk
    else:
        return "Low"


# =====================================================
# 3. CREATE SAMPLE
# =====================================================

def create_sample(pH, TDS, Turbidity, MP):

    ph_c = get_category(pH, "pH")
    tds_c = get_category(TDS, "TDS")
    turb_c = get_category(Turbidity, "Turbidity")
    mp_c = get_category(MP, "MP")

    risk = assign_risk(
        ph_c,
        tds_c,
        turb_c,
        mp_c
    )

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
# 4. BOUNDARY VALUES
# =====================================================

# These values specifically test both sides
# of every important boundary.

pH_boundary = [
    6.4,
    6.5,
    7.0,
    7.1,
    7.8,
    7.9,
    8.5,
    8.6
]

TDS_boundary = [
    149,
    150,
    199,
    200,
    270,
    271,
    350,
    351
]

Turbidity_boundary = [
    0.9,
    1.0,
    4.9,
    5.0
]

MP_boundary = [
    0,
    1,
    5,
    6
]


# =====================================================
# 5. CREATE BOUNDARY DATASET
# =====================================================

boundary_data = []

print("\nCreating boundary cases...")

# Keep other parameters at safe middle values
# while testing each parameter boundary.

for value in pH_boundary:

    boundary_data.append(
        create_sample(
            value,
            230,
            2.0,
            2
        )
    )


for value in TDS_boundary:

    boundary_data.append(
        create_sample(
            7.4,
            value,
            2.0,
            2
        )
    )


for value in Turbidity_boundary:

    boundary_data.append(
        create_sample(
            7.4,
            230,
            value,
            2
        )
    )


for value in MP_boundary:

    boundary_data.append(
        create_sample(
            7.4,
            230,
            2.0,
            value
        )
    )


# =====================================================
# 6. CATEGORY REPRESENTATIVE VALUES
# =====================================================

# These values make sure every category is represented.

pH_values = [
    6.4,   # Poor
    6.7,   # Moderate
    7.4,   # Good
    8.2,   # Moderate
    8.7    # Poor
]

TDS_values = [
    100,   # Poor
    175,   # Moderate
    230,   # Good
    300,   # Moderate
    400    # Poor
]

Turbidity_values = [
    0.5,   # Good
    2.5,   # Moderate
    5.0,   # Poor
    8.0     # Poor
]

MP_values = [
    0,      # Good
    3,      # Moderate
    6,      # Poor
    15      # Poor
]


# =====================================================
# 7. GENERATE RANDOM SAMPLES
# =====================================================

def generate_random_sample():

    pH = round(
        np.random.uniform(4.5, 10.0),
        1
    )

    TDS = round(
        np.random.uniform(50, 800),
        1
    )

    Turbidity = round(
        np.random.uniform(0, 15),
        1
    )

    MP = np.random.randint(0, 30)

    return create_sample(
        pH,
        TDS,
        Turbidity,
        MP
    )


# =====================================================
# 8. GENERATE BALANCED DATASET
# =====================================================

data = []

print("\nGenerating balanced dataset...")


# Generate balanced risk classes
while len(data) < 1500:

    sample = generate_random_sample()

    # Count current classes
    current_risks = [
        row["Risk"]
        for row in data
    ]

    low_count = current_risks.count("Low")
    medium_count = current_risks.count("Medium")
    high_count = current_risks.count("High")

    # Keep each class close to 500
    if sample["Risk"] == "Low" and low_count < 500:
        data.append(sample)

    elif sample["Risk"] == "Medium" and medium_count < 500:
        data.append(sample)

    elif sample["Risk"] == "High" and high_count < 500:
        data.append(sample)


# =====================================================
# 9. ADD BOUNDARY CASES
# =====================================================

# Replace some random samples with boundary cases
# so the final dataset remains exactly 1500 rows.

data[-len(boundary_data):] = boundary_data

df = pd.DataFrame(data)


# =====================================================
# 10. SHUFFLE DATASET
# =====================================================

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# =====================================================
# 11. CHECK CATEGORY COUNTS
# =====================================================

print("\n==============================")
print("CATEGORY DISTRIBUTION")
print("==============================")

print("\npH:")
print(df["pH_cat"].value_counts())

print("\nTDS:")
print(df["TDS_cat"].value_counts())

print("\nTurbidity:")
print(df["Turbidity_cat"].value_counts())

print("\nMicroplastics:")
print(df["MP_cat"].value_counts())

print("\nRisk:")
print(df["Risk"].value_counts())


# =====================================================
# 12. SAVE DATASET
# =====================================================

df.to_csv(
    "datasets/water_quality_dataset.csv",
    index=False
)
