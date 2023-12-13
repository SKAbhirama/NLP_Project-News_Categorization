# Feature Extraction
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load your lemmatized data from Excel
lemmatized_data_path = '/content/lemmatized_10k.xlsx'
df_lemmatized = pd.read_excel(lemmatized_data_path)

# Load the 'News Categories' data from another Excel sheet
news_categories_path = '/content/updated.xlsx'
df_categories = pd.read_excel(news_categories_path)

# Function to clean 'News Categories' column
def clean_categories(category):
    if isinstance(category, str):
        # Remove non-alphabetic characters and truncate text after a comma
        cleaned_category = ''.join(filter(str.isalpha, category.split(',')[0]))
        return cleaned_category.lower()  # Convert to lowercase for consistency
    return category

# Clean the 'News Categories' column
df_categories['News Categories'] = df_categories['News Categories'].apply(clean_categories)

# Split the data into batches of 4 rows
batch_size = 2500
num_batches = len(df_lemmatized) // batch_size

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# List to store TF-IDF matrices for each batch
tfidf_matrices = []

# Process data in batches
for batch_num in range(num_batches):
    start_idx = batch_num * batch_size
    end_idx = (batch_num + 1) * batch_size

    # Extract the current batch
    current_batch = df_lemmatized['Headline'].iloc[start_idx:end_idx]

    # Fit and transform the current batch
    tfidf_batch = vectorizer.fit_transform(current_batch)

    # Store the TF-IDF matrix in the list
    tfidf_matrices.append(tfidf_batch)

# Concatenate all TF-IDF matrices into one matrix
all_tfidf_matrices = pd.concat([pd.DataFrame(matrix.toarray()) for matrix in tfidf_matrices], ignore_index=True)

# Add the 'News Categories' column to the final DataFrame
all_tfidf_matrices['Category'] = df_categories['News Categories']

vectorizer_path = '/content/vectorizer.joblib'
joblib.dump(vectorizer, vectorizer_path)

# Save the final DataFrame to a CSV file
all_tfidf_matrices.to_csv('/content/final_tfidf_output1.csv', index=False)

all_tfidf_matrices.head()
