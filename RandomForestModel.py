from preprocessing import load_and_preprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

# 1. Load preprocessed data
X_train_scaled, X_test_scaled, y_train, y_test = load_and_preprocess()

# 2. Initialize Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)

# 3. Train the model
rf.fit(X_train_scaled, y_train)

# 4. Predict on test set
y_pred = rf.predict(X_test_scaled)

# 5. Evaluation
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
plt.show()

# 7. Feature Importance
features = ["Systolic_BP", "Diastolic_BP", "BMI Category",
            "Daily Steps", "Sleep Duration", "Occupation"]

importances = rf.feature_importances_
fi_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=False)
print("\nFeature Importance:\n", fi_df)
