from preprocessing import load_and_preprocess
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

# 1. Load preprocessed numeric data
X_train_scaled, X_test_scaled, y_train, y_test = load_and_preprocess()

# 2. Initialize CatBoost
cat_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=4,
    eval_metric='Accuracy',
    random_seed=42,
    verbose=100
)

# 3. Train CatBoost (no cat_features needed)
cat_model.fit(X_train_scaled, y_train)

# 4. Predict
y_pred = cat_model.predict(X_test_scaled)

# 5. Evaluation
print("CatBoost Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.title("CatBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
plt.show()

# 7. Feature Importance
features = ["Systolic_BP", "Diastolic_BP", "BMI Category",
            "Daily Steps", "Sleep Duration", "Occupation"]
fi_df = pd.DataFrame({"Feature": features, "Importance": cat_model.get_feature_importance()})
fi_df = fi_df.sort_values(by="Importance", ascending=False)
print("\nFeature Importance:\n", fi_df)
