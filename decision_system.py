import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("student_decision_data.csv")

# Separate features and target
X = df.drop("risk_level", axis=1)
y = df["risk_level"]

# Encode participation level
encoder = LabelEncoder()
X["participation_level"] = encoder.fit_transform(X["participation_level"])

# Train decision tree model
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X, y)

# ---------------- USER INPUT ----------------
print("\n=== Student Decision Intelligence System ===\n")

attendance = float(input("Attendance percentage: "))
study_hours = float(input("Average study hours per day: "))
previous_marks = float(input("Previous marks: "))
assignments = int(input("Assignments completed: "))
test_score = float(input("Recent test score: "))
participation = input("Participation level (Low / Medium / High): ")

# Encode user participation input
participation_encoded = encoder.transform([participation])[0]

# Prepare input for model
user_data = [[
    attendance,
    study_hours,
    previous_marks,
    assignments,
    test_score,
    participation_encoded
]]

# Predict risk
risk = model.predict(user_data)[0]
print("\nPredicted Risk Level:", risk)

# ---------------- EXPLANATION ENGINE ----------------
feature_importance = model.feature_importances_
features = X.columns

importance_map = dict(zip(features, feature_importance))
sorted_factors = sorted(importance_map.items(), key=lambda x: x[1], reverse=True)

print("\nTop contributing factors:")
for factor, score in sorted_factors[:3]:
    print(f"- {factor.replace('_',' ').title()} ({score:.2f})")

# ---------------- DECISION SUGGESTIONS ----------------
print("\nSuggested Actions:")

if attendance < 75:
    print("- Improve attendance through mentoring")

if study_hours < 2.5:
    print("- Create a structured daily study plan")

if assignments < 7:
    print("- Encourage timely assignment submission")

if participation == "Low":
    print("- Increase classroom engagement activities")

print("\n===========================================")
