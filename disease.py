import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report


# Load dataset
data = pd.read_csv("D:\\training1.csv")
print(data.dropna())

# Separate input and output
x = data.drop("prognosis", axis=1)
y = data["prognosis"]


# Encode disease names
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=42,
)


# ==================================================
# DECISION TREE
# ==================================================

dt = DecisionTreeClassifier(random_state=42)

dt.fit(x_train, y_train)

dt_pred = dt.predict(x_test)

dt_accuracy = accuracy_score(y_test, dt_pred)


# ==================================================
# RANDOM FOREST
# ==================================================

rf = RandomForestClassifier(
    random_state=42
)

rf.fit(x_train, y_train)

rf_pred = rf.predict(x_test)

rf_accuracy = accuracy_score(y_test, rf_pred)


# ==================================================
# KNN
# ==================================================

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train, y_train)

knn_pred = knn.predict(x_test)

knn_accuracy = accuracy_score(y_test, knn_pred)


# ==================================================
# LOGISTIC REGRESSION
# ==================================================

lr = LogisticRegression(max_iter=1000)

lr.fit(x_train, y_train)

lr_pred = lr.predict(x_test)

lr_accuracy = accuracy_score(y_test, lr_pred)


# ==================================================
# ACCURACY
# ==================================================

print("\nMODEL ACCURACY")

print("Decision Tree:", dt_accuracy)

print("Random Forest:", rf_accuracy)

print("KNN:", knn_accuracy)

print("Logistic Regression:", lr_accuracy)


# ==================================================
# CLASSIFICATION REPORT - DECISION TREE
# ==================================================

print("\n==========================================")
print("DECISION TREE CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        y_test,
        dt_pred
        
    )
)


# ==================================================
# CLASSIFICATION REPORT - RANDOM FOREST
# ==================================================

print("\n==========================================")
print("RANDOM FOREST CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        y_test,
        rf_pred
    )
)


# ==================================================
# CLASSIFICATION REPORT - KNN
# ==================================================

print("\n==========================================")
print("KNN CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        y_test,
        knn_pred
    )
)


# ==================================================
# CLASSIFICATION REPORT - LOGISTIC REGRESSION
# ==================================================

print("\n==========================================")
print("LOGISTIC REGRESSION CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        y_test,
        lr_pred
    )
)


# ==================================================
# FOUR MODELS COMPARISON
# ==================================================

results = pd.DataFrame({
    "Model": [
        "Decision Tree",
        "Random Forest",
        "KNN",
        "Logistic Regression"
    ],
    "Accuracy": [
        dt_accuracy,
        rf_accuracy,
        knn_accuracy,
        lr_accuracy
    ]
})

print("\n==========================================")
print("FOUR MODELS COMPARISON")
print("==========================================")

print(results.to_string(index=False))

new_patient = pd.DataFrame([[
    0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]], columns=x.columns)

print("\nDISEASE PREDICTION RESULTS")
print("Decision Tree:", label_encoder.inverse_transform(dt.predict(new_patient))[0])
print("Random Forest:", label_encoder.inverse_transform(rf.predict(new_patient))[0])
print("KNN:", label_encoder.inverse_transform(knn.predict(new_patient))[0])
print("Logistic Regression:", label_encoder.inverse_transform(lr.predict(new_patient))[0])



# ==================================================
# BAR CHART
# ==================================================

models = [
    "Decision Tree",
    "Random Forest",
    "KNN",
    "Logistic Regression"
]

accuracies = [
    dt_accuracy,
    rf_accuracy,
    knn_accuracy,
    lr_accuracy
]

plt.figure(figsize=(8, 5))

plt.bar(models, accuracies)

plt.xlabel("Models")
plt.ylabel("Accuracy")

plt.title("Disease Prediction Model Comparison")


plt.show()
