import joblib
import numpy as np
import pandas as pd

# Load saved model and preprocessing tools
gb_model = joblib.load("gb_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

feature_names = ["Systolic_BP", "Diastolic_BP", "BMI Category",
                 "Daily Steps", "Sleep Duration", "Occupation"]

def get_user_input():
    print("Enter new data for prediction:")

    systolic = float(input("Systolic BP (e.g., 120): "))
    diastolic = float(input("Diastolic BP (e.g., 80): "))

    print("BMI Categories:", list(encoders["BMI Category"].classes_))
    bmi_cat = input("BMI Category (case-sensitive): ")
    while bmi_cat not in encoders["BMI Category"].classes_:
        print("Invalid input. Try again.")
        bmi_cat = input("BMI Category (case-sensitive): ")

    steps = int(input("Daily Steps (e.g., 5000): "))
    sleep_duration = float(input("Sleep Duration (hours, e.g., 7.5): "))

    print("Occupations:", list(encoders["Occupation"].classes_))
    occupation = input("Occupation (case-sensitive): ")
    while occupation not in encoders["Occupation"].classes_:
        print("Invalid input. Try again.")
        occupation = input("Occupation (case-sensitive): ")

    return {
        "Systolic_BP": systolic,
        "Diastolic_BP": diastolic,
        "BMI Category": bmi_cat,
        "Daily Steps": steps,
        "Sleep Duration": sleep_duration,
        "Occupation": occupation
    }

def preprocess_input(data):
    bmi_encoded = encoders["BMI Category"].transform([data["BMI Category"]])[0]
    occupation_encoded = encoders["Occupation"].transform([data["Occupation"]])[0]

    # Create DataFrame with feature names to satisfy scaler
    df = pd.DataFrame([[
        data["Systolic_BP"],
        data["Diastolic_BP"],
        bmi_encoded,
        data["Daily Steps"],
        data["Sleep Duration"],
        occupation_encoded
    ]], columns=feature_names)

    features_scaled = scaler.transform(df)
    return features_scaled

def main():
    data = get_user_input()
    X = preprocess_input(data)
    pred_encoded = gb_model.predict(X)[0]
    pred_label = target_encoder.inverse_transform([pred_encoded])[0]
    print(f"\nPredicted Sleep Disorder: {pred_label}")

if __name__ == "__main__":
    main()
