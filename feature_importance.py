from preprocessing import load_and_preprocess
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load preprocessed data
X_train_scaled, X_test_scaled, y_train, y_test = load_and_preprocess()

# 2. Train Gradient Boosting on original data
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train_scaled, y_train)

# 3. Feature names
features = ["Systolic_BP", "Diastolic_BP", "BMI Category",
            "Daily Steps", "Sleep Duration", "Occupation"]

# 4. Get feature importances
importances = gb.feature_importances_
fi_df = pd.DataFrame({"Feature": features, "Importance": importances})
fi_df = fi_df.sort_values(by="Importance", ascending=False)

# 5. Print feature importance
print(fi_df)

# 6. Plot feature importance
plt.figure(figsize=(8,5))
plt.barh(fi_df['Feature'], fi_df['Importance'], color='skyblue')
plt.xlabel("Importance")
plt.title("Feature Importance - Gradient Boosting")
plt.gca().invert_yaxis()  # Highest importance on top
plt.show()
