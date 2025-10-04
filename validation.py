from preprocessing import load_and_preprocess
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# 1. Load preprocessed data
X_train_scaled, X_test_scaled, y_train, y_test = load_and_preprocess()

# 2. Combine train and test to use full dataset for cross-validation
import numpy as np
X_full = np.vstack((X_train_scaled, X_test_scaled))
y_full = np.concatenate((y_train, y_test))

# 3. Initialize Gradient Boosting
gb = GradientBoostingClassifier(random_state=42)

# 4. Perform 5-fold cross-validation
cv_scores = cross_val_score(gb, X_full, y_full, cv=5, scoring='accuracy')

# 5. Print results
print("Cross-Validation Accuracy Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())
print("Standard Deviation:", cv_scores.std())
