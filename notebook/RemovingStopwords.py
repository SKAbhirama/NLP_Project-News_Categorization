# Code to remove stopwords
# Download the NLTK stop words dataset
nltk.download('stopwords')

# Load the Excel file into a pandas DataFrame
excel_path = '/english_news_dataset.xlsx'  # Update with your file path
df = pd.read_excel(excel_path)

# Specify the columns containing the text data
column_name1 = 'Headline'  # Replace with your column name
column_name2 = 'Content'

# Function to remove special characters using regular expressions
def remove_special_characters(text):
    # Replace quotes and hyphens with white spaces
    text = re.sub(r'[\'\"-]', ' ', str(text))
    # Keep only alphanumeric characters and spaces
    pattern = r'[^a-zA-Z0-9\s]'
    return re.sub(pattern, '', text)

# Function to remove stop words
def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(str(text))
    filtered_text = ' '.join([word for word in words if word.lower() not in stop_words])
    return filtered_text

# Apply the functions to the specified columns
df[column_name1] = df[column_name1].apply(remove_special_characters).apply(remove_stop_words)
df[column_name2] = df[column_name2].apply(remove_special_characters).apply(remove_stop_words)

# Save the DataFrame to a new Excel file
output_excel_path = '/content/updated.xlsx'  # Update with your desired output file path
df.to_excel(output_excel_path, index=False)

# Display the modified DataFrame
df.head()
