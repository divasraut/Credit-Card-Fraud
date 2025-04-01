# Credit-Card-Fraud
This notebook focuses on credit card fraud detection using oversampling techniques to handle the imbalanced dataset
Credit card fraud detection is a binary classification problem, where the dataset has many more legitimate transactions (Class 0) than fraudulent ones (Class 1). To improve the model’s performance, oversampling techniques are applied to balance the dataset before training a machine learning model.

The key steps in this notebook include:

Loading and exploring the dataset to understand its structure.

Checking for missing values and dataset imbalance.

Applying oversampling techniques to balance the dataset.

Training a machine learning model (e.g., Logistic Regression, Decision Tree, etc.).

Evaluating the model's performance using accuracy, precision, recall, and F1-score.

Step-by-Step Breakdown
Step 1: Importing Required Libraries
python
Copy
Edit
import pandas as pd
import numpy as np
Purpose:

pandas → For handling and analyzing datasets.

numpy → For numerical computations.

Step 2: Loading the Credit Card Fraud Dataset
python
Copy
Edit
df = pd.read_csv('C:\\Users\\Divas Raut\\Desktop\\projects from netzwerk\\25_Projects\\ML\\Creadi Card Fraud Problem Handling Imbalanced Dataset\\creditcard.csv')
Purpose:

Loads the credit card transactions dataset from a CSV file.

The dataset includes transaction details, which will be used for fraud detection.

Step 3: Displaying the First Few Rows
python
Copy
Edit
df.head()
Purpose:

Displays the first 5 rows to inspect the dataset structure.

Step 4: Checking for Missing Values
python
Copy
Edit
df.isnull().sum()
Purpose:

Identifies missing values in the dataset.

Helps in data cleaning before model training.

Step 5: Understanding Dataset Information
python
Copy
Edit
df.info()
Purpose:

Shows column types, non-null counts, and memory usage.

Helps identify potential data cleaning steps.

Step 6: Checking Dataset Shape
python
Copy
Edit
df.shape
Purpose:

Returns the number of rows and columns.

Step 7: Checking for Class Imbalance
python
Copy
Edit
df.groupby(['Class']).count()
Purpose:

Displays the count of legitimate (Class 0) and fraudulent (Class 1) transactions.

Identifies if fraud cases are significantly fewer than non-fraud cases.

Step 8: Splitting Features and Labels
python
Copy
Edit
x = df.drop(['Time','Class'], axis=1).values
y = df.iloc[:, -1]
Purpose:

x: Stores features (excluding Time and Class).

y: Stores target labels (fraud or non-fraud).

Step 9: Splitting Data into Training and Testing Sets
python
Copy
Edit
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
Purpose:

Splits the dataset into 80% training and 20% testing.

Ensures fair evaluation of the model.

Step 10: Training a Machine Learning Model
python
Copy
Edit
from sklearn.linear_model import LinearRegression
model = LinearRegression()
⚠ Issue:

Linear Regression is not suitable for classification tasks.

Instead, Logistic Regression or another classification model should be used.

Step 11: Applying Oversampling to Handle Class Imbalance
python
Copy
Edit
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=0)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
Purpose:

SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic fraudulent transactions to balance the dataset.

Prevents the model from being biased towards the majority class (legitimate transactions).

Step 12: Training a Classification Model (After Oversampling)
python
Copy
Edit
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train_resampled, y_train_resampled)
Purpose:

Trains a Random Forest Classifier, which is robust against imbalanced data.

Uses the resampled dataset to improve fraud detection.

Step 13: Making Predictions
python
Copy
Edit
y_pred = model.predict(x_test)
Purpose:

Predicts whether a transaction is fraudulent or legitimate.

Step 14: Evaluating Model Performance
python
Copy
Edit
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
Purpose:

Generates Precision, Recall, and F1-score to assess model performance.

Summary of Steps
Loaded the credit card fraud dataset.

Checked for missing values and dataset structure.

Identified class imbalance in fraud vs. non-fraud transactions.

Split data into training and testing sets.

Applied SMOTE to balance the dataset.

Trained a Random Forest Classifier.

Made predictions on the test set.

Evaluated the model using classification metrics.

Final Thoughts
This project aims to improve credit card fraud detection by using oversampling (SMOTE) to handle class imbalance and training a Random Forest model for classification. The key takeaway is that handling imbalanced datasets is crucial for fraud detection, as ignoring it can lead to a model that fails to detect fraudulent transactions effectively.
