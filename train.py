from preprocess import load_and_preprocess
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib

# 1. Load preprocessed data and tools
X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoders, target_encoder = load_and_preprocess()

# 2. Train Gradient Boosting model
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train_scaled, y_train)

# 3. Predict on test data
y_pred = gb.predict(X_test_scaled)

# 4. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 5. Confusion Matrix visualization
cm = confusion_matrix(y_test, y_pred)
class_names = target_encoder.classes_

plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(ticks=range(len(class_names)), labels=class_names)
plt.yticks(ticks=range(len(class_names)), labels=class_names)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="red")

plt.show()

# 6. Save model and preprocessing tools
joblib.dump(gb, "gb_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")

print("✅ Model and preprocessing tools saved successfully.")
