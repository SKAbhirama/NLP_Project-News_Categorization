import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the TF-IDF matrix and 'News Categories' from CSV
tfidf_matrix_path = '/content/final_tfidf_output.csv'
df_tfidf = pd.read_csv(tfidf_matrix_path)

# Drop samples with NaN values
df_tfidf_no_nan = df_tfidf.dropna()

# Split the data into features (X) and target labels (y)
X = df_tfidf_no_nan.drop('Category', axis=1)
y = df_tfidf_no_nan['Category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the SVM classifier
svm_classifier = SVC()

# Train the SVM classifier on the training data
svm_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = svm_classifier.predict(X_test)

# Evaluate the performance of the classifier
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report_str)
