import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess():
    # 1. Load Dataset
    df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

    # 2. Drop irrelevant and missing values
    df = df.drop(columns=["Person ID"])
    df = df.dropna(subset=["Sleep Disorder", "Blood Pressure", "BMI Category",
                           "Daily Steps", "Sleep Duration", "Occupation"])

    # 3. Feature Engineering: Split Blood Pressure
    bp_split = df["Blood Pressure"].str.split("/", expand=True)
    df["Systolic_BP"] = bp_split[0].astype(float)
    df["Diastolic_BP"] = bp_split[1].astype(float)
    df = df.drop(columns=["Blood Pressure"])

    # 4. Select Features and Target
    features = ["Systolic_BP", "Diastolic_BP", "BMI Category",
                "Daily Steps", "Sleep Duration", "Occupation"]
    X = df[features].copy()
    y = df["Sleep Disorder"].copy()

    # 5. Encoding categorical columns with separate encoders
    encoders = {}
    for col in ["BMI Category", "Occupation"]:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    # Encode target
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)

    # 6. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # 7. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Return all required objects
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoders, target_encoder
