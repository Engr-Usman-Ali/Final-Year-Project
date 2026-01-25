import random
import pandas as pd
import os

# =====================================================
# SECTION 1: PARAMETER CATEGORIZATION FUNCTIONS
# =====================================================

def ph_category(ph):
    ph = round(ph, 2)    # Round to 2 decimals for consistency
    if 7.1 <= ph <= 7.8:
        return "Good"
    elif (6.5 <= ph <= 6.9) or (7.9 <= ph <= 8.5):
        return "Moderate"
    else:
        return "Poor"  

def tds_category(tds):
    if 200 <= tds <= 270:
        return "Good"
    elif 150 <= tds <= 199 or 271 <= tds <= 350:
        return "Moderate"
    else:
        return "Poor"

def turbidity_category(turb):
    if turb < 1:
        return "Good"
    elif 1 <= turb < 5:
        return "Moderate"
    else:
        return "Poor"

def mp_category(mp):
    if mp <= 4:
        return "Good"
    elif 5 <= mp <= 7:
        return "Moderate"
    else:
        return "Poor"

# =====================================================
# SECTION 2: RISK FUSION LOGIC
# =====================================================

def calculate_risk(ph_c, tds_c, turb_c, mp_c):
    categories = [ph_c, tds_c, turb_c, mp_c]
    if "Poor" in categories:
        return "High"
    if categories.count("Moderate") >= 2:
        return "Medium"
    return "Low"

# =====================================================
# SECTION 3: CONTROLLED VALUE GENERATORS
# =====================================================

def generate_good_values():
    ph = round(random.uniform(7.1, 7.8), 2)  
    tds = random.randint(200, 270)
    turb = round(random.uniform(0.0, 0.9), 2)
    mp = random.randint(0, 4)
    return ph, tds, turb, mp

def generate_moderate_values():
    ph = round(random.choice([random.uniform(6.5, 6.9), random.uniform(7.9, 8.5)]), 2)
    tds = random.choice([random.randint(150, 199), random.randint(271, 350)])
    turb = round(random.uniform(1.0, 4.9), 2)
    mp = random.randint(5, 7)
    return ph, tds, turb, mp

def generate_poor_values():
    ph = round(random.choice([
        7.0,  
        random.uniform(1.0, 6.4),
        random.uniform(8.6, 14.0)
    ]), 2)
    tds = random.choice([random.randint(50, 149), random.randint(351, 1000)])
    turb = round(random.uniform(5.1, 20.0), 2)
    mp = random.randint(8, 50)
    return ph, tds, turb, mp

# =====================================================
# SECTION 4: GENERATE DATASET (FIXED INDENTATION)
# =====================================================

data = []
ROWS_PER_CLASS = 1700

# ---- LOW RISK ----
for _ in range(ROWS_PER_CLASS):
    ph, tds, turb, mp = generate_good_values()
    data.append([ph, tds, turb, mp, "Low"])

# ---- MEDIUM RISK ----
for _ in range(ROWS_PER_CLASS):
    # Start with all good values
    vals = list(generate_good_values())
    mod_vals = list(generate_moderate_values())
    
    # Force exactly 2 or 3 parameters to be moderate
    num_to_change = random.randint(2, 3)
    indices = random.sample(range(4), num_to_change)
    for idx in indices:
        vals[idx] = mod_vals[idx]
    
    data.append(vals + ["Medium"])

# ---- HIGH RISK ----
for _ in range(ROWS_PER_CLASS):
    # Start with all good values
    vals = list(generate_good_values())
    poor_vals = list(generate_poor_values())
    
    # Any single Poor value results in High Risk
    num_poor = random.randint(1, 4)
    indices = random.sample(range(4), num_poor)
    
    for idx in indices:
        vals[idx] = poor_vals[idx]
        
    data.append(vals + ["High"])

# =====================================================
# SECTION 5: CREATE DATAFRAME & SAVE CSV
# =====================================================

# Ensure directory exists
if not os.path.exists("./datasets"):
    os.makedirs("./datasets")

df = pd.DataFrame(data, columns=["pH", "TDS", "Turbidity", "MP_Count", "Risk"])

# Shuffle the dataset so the model doesn't see chunks of the same class
df = df.sample(frac=1).reset_index(drop=True)

df.to_csv("./datasets/drinking_water_dataset.csv", index=False)

print("-" * 30)
print("Dataset Created Successfully!")
print("\nReference Standards Used:")
print("• Pakistan Council of Research in Water Resources (PCRWR), Pakistan")
print("• Environmental Protection Agency, Azad Jammu & Kashmir (EPA-AJK)")
print("• Muslim Hands International – Water, Sanitation & Hygiene (WASH) Program")
print("-" * 30)
