import pandas as pd
import spacy

# Load spaCy's English language model
nlp = spacy.load('en_core_web_sm')

# Load the Excel file into a pandas DataFrame
excel_path = '/content/updated.xlsx'  # Update with your file path
df = pd.read_excel(excel_path)
subset_df = df.head(10000)

# Specify the columns containing the text data
column_name1 = 'Headline'  # Replace with your column name
column_name2 = 'Content'

# Function to apply lemmatization
def lemmatize_text(text):
    doc = nlp(str(text))
    lemmatized_text = ' '.join([token.lemma_ for token in doc])
    return lemmatized_text

# Apply the lemmatization function to the specified columns
subset_df[column_name1] = subset_df[column_name1].apply(lemmatize_text)
subset_df[column_name2] = subset_df[column_name2].apply(lemmatize_text)

# Save the DataFrame to a new Excel file
output_excel_path = '/content/lemmatized_10k.xlsx'  # Update with your desired output file path
subset_df.to_excel(output_excel_path, index=False)

# Display the first 5 rows of the modified DataFrame
subset_df.head()
