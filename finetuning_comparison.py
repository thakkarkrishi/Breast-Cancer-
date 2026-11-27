
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load Dataset
df = pd.read_csv("D:/data science/breast cancer/breastcancer.csv.csv")

# Remove Unnecessary Columns
df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)

# Handle Missing Values
df = df.dropna()


# Separate Input and Output
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Label Encoding
le = LabelEncoder()
y = le.fit_transform(y)


# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)


# Original Models (Without Fine Tuning)

original_models = {

    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    ),

    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "SVM": make_pipeline(
        StandardScaler(),
        SVC()
    ),

    "Naive Bayes": GaussianNB()
}


# Fine Tuning Models

tuned_models = {

    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "criterion": ["gini", "entropy"]
        }
    },

    "Logistic Regression": {
        "model": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000)
        ),
        "params": {
            "logisticregression__C": [0.01, 0.1, 1, 10, 100],
            "logisticregression__solver": ["liblinear", "lbfgs"]
        }
    },

    "KNN": {
        "model": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier()
        ),
        "params": {
            "kneighborsclassifier__n_neighbors": [3, 5, 7, 9, 11, 15, 19],
            "kneighborsclassifier__weights": ["uniform", "distance"],
            "kneighborsclassifier__p": [1, 2]
        }
    },

    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10, 15],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"]
        }
    },

    "SVM": {
        "model": make_pipeline(
            StandardScaler(),
            SVC()
        ),
        "params": {
            "svc__C": [0.1, 1, 10, 100],
            "svc__kernel": ["linear", "rbf"],
            "svc__gamma": ["scale", "auto"]
        }
    },

    "Naive Bayes": {
        "model": GaussianNB(),
        "params": {
            "var_smoothing": np.logspace(-12, -7, 6)
        }
    }
}


# Evaluation Function

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0)
    }


# Train Original Models

original_results = {}

for name, model in original_models.items():

    print("\nTraining Original:", name)

    model.fit(X_train, y_train)

    original_results[name] = evaluate_model(
        model, X_test, y_test
    )


# Fine Tuning

tuned_results = {}
best_models = {}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

for name, details in tuned_models.items():

    print("\nFine Tuning:", name)

    grid = GridSearchCV(
        details["model"],
        details["params"],
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    best_models[name] = best_model

    tuned_results[name] = evaluate_model(
        best_model, X_test, y_test
    )

    print("Best Parameters:", grid.best_params_)
    print("Best CV Accuracy:", grid.best_score_)


# Create Comparison DataFrame

comparison = []

for name in original_models:

    for metric in ["Accuracy", "Precision", "Recall", "F1 Score"]:

        before = original_results[name][metric]
        after = tuned_results[name][metric]

        comparison.append({
            "Model": name,
            "Metric": metric,
            "Without Fine Tuning": before,
            "With Fine Tuning": after,
            "Improvement": after - before
        })


comparison_df = pd.DataFrame(comparison)

print("\nComplete Model Comparison:")
print(comparison_df.round(4))


# Accuracy Comparison

accuracy_comparison = pd.DataFrame({

    "Model": list(original_models.keys()),

    "Without Fine Tuning": [
        original_results[name]["Accuracy"]
        for name in original_models
    ],

    "With Fine Tuning": [
        tuned_results[name]["Accuracy"]
        for name in original_models
    ]

})

print("\nAccuracy Comparison:")
print(accuracy_comparison.round(4))


# Accuracy Graph

accuracy_plot = accuracy_comparison.melt(
    id_vars="Model",
    var_name="Tuning",
    value_name="Accuracy"
)

plt.figure(figsize=(11, 6))

sns.barplot(
    data=accuracy_plot,
    x="Model",
    y="Accuracy",
    hue="Tuning"
)

plt.title("Before vs After Fine Tuning - Accuracy")
plt.xticks(rotation=30)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()


# Metric Comparison Graph

metric_df = comparison_df.melt(
    id_vars=["Model", "Metric"],
    value_vars=["Without Fine Tuning", "With Fine Tuning"],
    var_name="Tuning",
    value_name="Score"
)

plt.figure(figsize=(14, 7))

sns.barplot(
    data=metric_df,
    x="Model",
    y="Score",
    hue="Tuning"
)

plt.title("Classification Performance Before vs After Fine Tuning")
plt.xticks(rotation=30)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()


# Best Model Comparison

best_original = max(
    original_results,
    key=lambda x: original_results[x]["Accuracy"]
)

best_tuned = max(
    tuned_results,
    key=lambda x: tuned_results[x]["Accuracy"]
)

print("\nBest Original Model:", best_original)
print("Original Accuracy:", original_results[best_original]["Accuracy"])

print("\nBest Fine Tuned Model:", best_tuned)
print("Fine Tuned Accuracy:", tuned_results[best_tuned]["Accuracy"])


# Save Comparison Results

comparison_df.to_csv(
    "D:/data science/breast cancer/fine_tuning_comparison.csv",
    index=False
)

# Save Fine Tuned Models

with open(
    "D:/data science/breast cancer/breast_cancer_models.pkl",
    "wb"
) as file:

    pickle.dump((best_models, le), file)


print("\nComparison Results Saved Successfully!")
