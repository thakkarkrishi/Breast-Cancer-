
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Dataset
df = pd.read_csv("D:/data science/breast cancer/breastcancer.csv.csv")

# Remove Unnecessary Columns
df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)

# Display Dataset
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

# Check Missing Values
print("Missing Values:")
print(df.isnull().sum())

# Handle Missing Values
df = df.dropna()

# Check Duplicate Values
print("Duplicate Values:", df.duplicated().sum())

# EDA

# Diagnosis Distribution
plt.figure(figsize=(6, 6))
df['diagnosis'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Diagnosis Distribution")
plt.ylabel("")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(15, 12))
sns.heatmap(df.select_dtypes(include=np.number).corr(), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

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

# Classification Models

models = {
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
        n_estimators=100, random_state=42
    ),

    "SVM": make_pipeline(
        StandardScaler(),
        SVC()
    ),

    "Naive Bayes": GaussianNB()
}

# Train and Evaluate Models

for name, model in models.items():

    print("\n", name)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("Classification Report:")
    print(classification_report(
        y_test, y_pred, target_names=le.classes_
    ))

# Accuracy Comparison

results = {}

for name, model in models.items():
    y_pred = model.predict(X_test)
    results[name] = accuracy_score(y_test, y_pred)

accuracy_df = pd.DataFrame(
    list(results.items()),
    columns=['Model', 'Accuracy']
)

print("\nModel Accuracy Comparison:")
print(accuracy_df)

sns.barplot(x='Model', y='Accuracy', data=accuracy_df)
plt.xticks(rotation=45)
plt.title("Classification Model Accuracy Comparison")
plt.show()

# Save Models and LabelEncoder
with open("D:/data science/breast cancer/breast_cancer_models.pkl", "wb") as file:
    pickle.dump((models, le), file)

print("Models and LabelEncoder Saved Successfully")