
# STEP 0: INSTALL LIBRARIES

!pip install seaborn

#  STEP 1: IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# STEP 2: CREATE SYNTHETIC DATA

np.random.seed(42)

n = 1000

df = pd.DataFrame({
    "study_hours": np.random.randint(1, 10, n),
    "attendance": np.random.randint(50, 100, n),
    "previous_marks": np.random.randint(40, 100, n),
    "assignments": np.random.randint(40, 100, n)
})

# Create target variable (Pass/Fail logic)
df["pass"] = (
    (df["study_hours"] > 4) &
    (df["attendance"] > 60) &
    (df["previous_marks"] > 50)
).astype(int)


# STEP 3: DATA PREVIEW

print("Dataset Preview:")
display(df.head())

print("\nClass Distribution:")
print(df["pass"].value_counts())


#  STEP 4: VISUALIZATION

sns.countplot(x='pass', data=df)
plt.title("Pass vs Fail Distribution")
plt.show()


# STEP 5: SPLIT DATA

X = df.drop("pass", axis=1)
y = df["pass"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#  STEP 6: TRAIN MODEL

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


#  STEP 7: PREDICTIONS

y_pred = model.predict(X_test)


#  STEP 8: EVALUATION

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


#  STEP 9: CONFUSION MATRIX

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# STEP 10: TEST WITH NEW STUDENT

new_student = pd.DataFrame({
    "study_hours": [3],
    "attendance": [65],
    "previous_marks": [55],
    "assignments": [60]
})

prediction = model.predict(new_student)
probability = model.predict_proba(new_student)

print("\nNew Student Prediction:")
print("Pass (1) / Fail (0):", prediction[0])
print("Probability:", probability)


#  STEP 11: SAVE MODEL

import joblib
joblib.dump(model, "student_model.pkl")

print("\nModel saved as student_model.pkl")