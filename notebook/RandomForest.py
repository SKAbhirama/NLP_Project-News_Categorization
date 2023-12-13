# RandomForest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

# Initialize the Random Forest classifier
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
random_forest_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = random_forest_classifier.predict(X_test)

# Evaluate the performance of the classifier
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report_str)

''' Load the original vectorizer used during training
vectorizer_path = '/content/vectorizer.joblib'  # Change this to the path where you saved your original vectorizer
vectorizer = joblib.load(vectorizer_path)

def get_tfidf_matrix(input_text, vectorizer):
    # Transform the input text using the TF-IDF vectorizer
    tfidf_matrix = vectorizer.transform(input_text)
    return tfidf_matrix

# New input news sample for prediction
new_input_news = ["Govt redefines 'terrorist act', includes threat to monetary stability"]

# Transform the new input news using the separate method
df_tfidf_new = get_tfidf_matrix(new_input_news, vectorizer)

# Make predictions on the new input news
y_new_pred = random_forest_classifier.predict(df_tfidf_new)

print("Predicted Category for New Input News:", y_new_pred)'''
