# =====================================================
# File: 1_generate_dataset.py
# Purpose: Generate synthetic water quality dataset
# Features: Raw values + engineered features (NO categorical flags)
# =====================================================

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

def generate_water_quality_dataset(n_samples=5000):
    """
    Generate synthetic water quality dataset with realistic distributions.
    Uses WHO, PCRWR, and EPA-AJK standards as reference.
    """
    
    data = []
    
    for _ in range(n_samples):
        # Decide risk level first (stratified sampling)
        risk_label = np.random.choice(
            ['Low', 'Medium', 'High'], 
            p=[0.50, 0.35, 0.15]  # 50% Low, 35% Medium, 15% High
        )
        
        # =====================================================
        # Generate parameters based on risk level
        # =====================================================
        
        if risk_label == 'Low':
            # Safe water - all parameters in Good range
            pH = np.random.uniform(7.1, 7.8)
            TDS = np.random.uniform(200, 270)
            Turbidity = np.random.uniform(0, 0.9)
            MP_Count = 0
            
            # Add some variation (10% of Low samples have 1 Moderate parameter)
            if np.random.random() < 0.10:
                choice = np.random.randint(0, 3)
                if choice == 0:
                    pH = np.random.uniform(6.5, 7.0)  # Moderate pH
                elif choice == 1:
                    TDS = np.random.uniform(271, 320)  # Moderate TDS
                else:
                    Turbidity = np.random.uniform(1.0, 2.5)  # Moderate Turbidity
        
        elif risk_label == 'Medium':
            # Moderate contamination - 2+ Moderate or 1 Moderate + edge cases
            num_moderate = np.random.randint(2, 4)  # 2-3 Moderate parameters
            
            # Start with Good values
            pH = np.random.uniform(7.1, 7.8)
            TDS = np.random.uniform(200, 270)
            Turbidity = np.random.uniform(0, 0.9)
            MP_Count = 0
            
            # Make 2-3 parameters Moderate
            params = ['pH', 'TDS', 'Turbidity', 'MP_Count']
            moderate_params = np.random.choice(params, size=num_moderate, replace=False)
            
            for param in moderate_params:
                if param == 'pH':
                    if np.random.random() < 0.5:
                        pH = np.random.uniform(6.5, 7.0)  # Low Moderate
                    else:
                        pH = np.random.uniform(7.9, 8.5)  # High Moderate
                elif param == 'TDS':
                    if np.random.random() < 0.5:
                        TDS = np.random.uniform(150, 199)  # Low Moderate
                    else:
                        TDS = np.random.uniform(271, 350)  # High Moderate
                elif param == 'Turbidity':
                    Turbidity = np.random.uniform(1.0, 4.9)  # Moderate
                elif param == 'MP_Count':
                    MP_Count = np.random.randint(1, 6)  # 1-5 particles (Moderate)
        
        else:  # High risk
            # Contaminated water - at least 1 Poor parameter
            num_poor = np.random.randint(1, 3)  # 1-2 Poor parameters
            
            # Start with Good values
            pH = np.random.uniform(7.1, 7.8)
            TDS = np.random.uniform(200, 270)
            Turbidity = np.random.uniform(0, 0.9)
            MP_Count = 0
            
            # Make 1-2 parameters Poor
            params = ['pH', 'TDS', 'Turbidity', 'MP_Count']
            poor_params = np.random.choice(params, size=num_poor, replace=False)
            
            for param in poor_params:
                if param == 'pH':
                    if np.random.random() < 0.5:
                        pH = np.random.uniform(4.5, 6.4)  # Too acidic
                    else:
                        pH = np.random.uniform(8.6, 10.0)  # Too alkaline
                elif param == 'TDS':
                    if np.random.random() < 0.5:
                        TDS = np.random.uniform(50, 149)  # Too low
                    else:
                        TDS = np.random.uniform(351, 800)  # Too high
                elif param == 'Turbidity':
                    Turbidity = np.random.uniform(5.0, 15.0)  # Poor (≥5)
                elif param == 'MP_Count':
                    MP_Count = np.random.randint(6, 30)  # Poor (≥6)
            
            # Add some Moderate parameters to Poor samples
            remaining_params = [p for p in params if p not in poor_params]
            if len(remaining_params) > 0 and np.random.random() < 0.7:
                moderate_param = np.random.choice(remaining_params)
                if moderate_param == 'pH' and 7.1 <= pH <= 7.8:
                    pH = np.random.uniform(6.5, 7.0)
                elif moderate_param == 'TDS' and 200 <= TDS <= 270:
                    TDS = np.random.uniform(271, 350)
                elif moderate_param == 'Turbidity' and Turbidity < 1:
                    Turbidity = np.random.uniform(1.0, 4.9)
                elif moderate_param == 'MP_Count' and MP_Count == 0:
                    MP_Count = np.random.randint(1, 6)
        
        # =====================================================
        # Calculate engineered features (NOT categorical flags!)
        # =====================================================
        
        # Feature 1: pH deviation from neutral
        pH_deviation = abs(pH - 7.5)
        
        # Feature 2: Normalized TDS
        TDS_normalized = TDS / 1000.0
        
        # Feature 3: Pollution index (composite score)
        pollution_index = (MP_Count / 10.0) + ((TDS - 200) / 800.0) + (Turbidity / 20.0)
        
        # Feature 4: Interaction between Turbidity and Microplastics
        Turbidity_MP_interaction = Turbidity * (MP_Count + 1)
        
        # Feature 5: Combined boundary risk (how close to boundaries)
        pH_boundary_risk = min(abs(pH - 6.5), abs(pH - 8.5)) / 2.0
        TDS_boundary_risk = min(abs(TDS - 150), abs(TDS - 350)) / 200.0
        
        # Store sample
        data.append({
            # Raw parameters
            'pH': round(pH, 2),
            'TDS': int(TDS),
            'Turbidity': round(Turbidity, 2),
            'MP_Count': int(MP_Count),
            
            # Engineered features (continuous, not categorical)
            'pH_deviation': round(pH_deviation, 3),
            'TDS_normalized': round(TDS_normalized, 3),
            'pollution_index': round(pollution_index, 3),
            'Turbidity_MP_interaction': round(Turbidity_MP_interaction, 2),
            'pH_boundary_risk': round(pH_boundary_risk, 3),
            'TDS_boundary_risk': round(TDS_boundary_risk, 3),
            
            # Target label
            'Risk': risk_label
        })
    
    return pd.DataFrame(data)


# =====================================================
# GENERATE AND SAVE DATASET
# =====================================================

if __name__ == "__main__":
    print("=" * 80)
    print("GENERATING WATER QUALITY DATASET")
    print("=" * 80)
    
    # Generate dataset
    df = generate_water_quality_dataset(n_samples=5000)
    
    # Display statistics
    print(f"\n✓ Generated {len(df)} samples")
    print(f"\nRisk Distribution:")
    print(df['Risk'].value_counts().sort_index())
    print(f"\n{df['Risk'].value_counts(normalize=True).sort_index() * 100}")
    
    print("\nParameter Statistics:")
    print(df[['pH', 'TDS', 'Turbidity', 'MP_Count']].describe())
    
    print("\nEngineered Features Statistics:")
    print(df[['pH_deviation', 'TDS_normalized', 'pollution_index', 'Turbidity_MP_interaction']].describe())
    
    # Save dataset
    import os
    os.makedirs("./datasets", exist_ok=True)
    df.to_csv("./datasets/water_quality_dataset.csv", index=False)
    print("\n✓ Dataset saved to: ./datasets/water_quality_dataset.csv")
    
    # Show sample
    print("\nSample rows:")
    print(df.head(10))
    
    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETE")
    print("=" * 80)