import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.combine import SMOTEENN

# =======================
# 1. Load Dataset
# =======================
df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

# Drop irrelevant column
df = df.drop(columns=["Person ID"])

# Drop rows with missing values in important columns
df = df.dropna(subset=["Sleep Disorder", "Blood Pressure", "BMI Category",
                       "Daily Steps", "Sleep Duration", "Occupation"])

# =======================
# 2. Feature Engineering
# =======================
# Split Blood Pressure into Systolic and Diastolic
bp_split = df["Blood Pressure"].str.split("/", expand=True)
df["Systolic_BP"] = bp_split[0].astype(float)
df["Diastolic_BP"] = bp_split[1].astype(float)
df = df.drop(columns=["Blood Pressure"])

# Select Top 5 Features
features = ["Systolic_BP", "Diastolic_BP", "BMI Category",
            "Daily Steps", "Sleep Duration", "Occupation"]
X = df[features].copy()
y = df["Sleep Disorder"].copy()

# =======================
# 3. Encoding
# =======================
enc = LabelEncoder()
for col in ["BMI Category", "Occupation"]:
   X = X.copy()
   X.loc[:, col] = enc.fit_transform(X[col])

y = LabelEncoder().fit_transform(y)  # Encode target labels

# =======================
# 4. Train-Test Split
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =======================
# 5. Handle Imbalance with SMOTEENN
# =======================
smote_enn = SMOTEENN(random_state=42)
X_res, y_res = smote_enn.fit_resample(X_train, y_train)

# =======================
# 6. Scaling
# =======================
scaler = StandardScaler()
X_res = scaler.fit_transform(X_res)
X_test = scaler.transform(X_test)

# =======================
# 7. Train Gradient Boosting
# =======================
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_res, y_res)

# =======================
# 8. Evaluation
# =======================
y_pred = gb.predict(X_test)

print(" Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))