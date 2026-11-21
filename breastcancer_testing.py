import pandas as pd
import pickle

# Load Saved Models and LabelEncoder
with open("D:/data science/breast cancer/breast_cancer_models.pkl", "rb") as file:
    models, le = pickle.load(file)

print("Models Loaded Successfully")

# Load Dataset
df = pd.read_csv("D:/data science/breast cancer/breastcancer.csv.csv")

df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)

# Separate Input and Output
X = df.drop('diagnosis', axis=1)

# New Data for Prediction
new_data = X.iloc[[0]]

# Prediction Using All Models
for name, model in models.items():

    prediction = model.predict(new_data)

    result = le.inverse_transform(prediction)

    print(name, "Prediction:", result[0])
