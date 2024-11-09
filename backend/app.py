import re
import string
import numpy as np
import pandas as pd
import nltk
import PyPDF2
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from io import BytesIO
from nltk.tokenize import sent_tokenize
from flask_cors import CORS

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# Load stopwords and initialize other resources
stopwords_list = stopwords.words('english')
english_stopset = list(set(stopwords_list).union({
    "things", "that's", "something", "take", "don't", "may", "want", "you're",
    "set", "might", "says", "including", "lot", "much", "said", "know",
    "good", "step", "often", "going", "thing", "think", "back", "actually",
    "better", "look", "find", "right", "example", "verb", "verbs"
}))
lemmer = WordNetLemmatizer()
vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2),
                             min_df=1, max_df=0.9, max_features=10000,
                             lowercase=True, stop_words=english_stopset)

# Global storage for document chunks and vectorized data
documents_clean = []
titles = []
df = pd.DataFrame()

def process_text(text):
    # Clean and process text
    document = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
    document = re.sub(r'@\w+', '', document)         # Remove mentions
    document = document.lower()                      # Convert to lowercase
    document = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', document)
    document = re.sub(r'[0-9]', '', document)
    document = re.sub(r'\s{2,}', ' ', document)
    lemmatized_text = ' '.join([lemmer.lemmatize(word) for word in document.split()])
    return lemmatized_text

def chunk_text(text, sentences_per_chunk=10):
    # Split the text into sentences
    sentences = sent_tokenize(text)
    # Group sentences into chunks of `sentences_per_chunk`
    return [' '.join(sentences[i:i + sentences_per_chunk]) for i in range(0, len(sentences), sentences_per_chunk)]

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    global documents_clean, titles, df
    # Retrieve PDF file, query, and top_n from the form data
    pdf_file = request.files.get('pdf')
    query = request.form.get('query', "")
    top_n = int(request.form.get('top_n', 5))

    if not pdf_file or not query:
        return jsonify({"error": "PDF and query are required"}), 400

    # Load and process PDF
    pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
    full_text = ''.join(page.extract_text() for page in pdf_reader.pages if page.extract_text())

    # Split the text into sentence chunks and process each one
    text_chunks = chunk_text(full_text)
    documents_clean.clear()  # Clear any previous data
    titles.clear()
    for i, chunk in enumerate(text_chunks):
        processed_text = process_text(chunk)
        documents_clean.append(processed_text)
        titles.append(f"Uploaded PDF Document - Chunk {i+1}")

    # Vectorize all chunks
    if documents_clean:
        try:
            vectorized_data = vectorizer.fit_transform(documents_clean)
            df = pd.DataFrame(vectorized_data.T.toarray())
        except ValueError as e:
            return jsonify({"error": str(e)}), 500

    # Process query and retrieve top_n results
    query_processed = process_text(query)
    try:
        query_vector = vectorizer.transform([query_processed]).toarray().flatten()
    except ValueError as e:
        return jsonify({"error": f"Vectorizer error: {str(e)}"}), 500

    # Ensure the query vector aligns with df
    if query_vector.shape[0] != df.shape[0]:  # Check if number of features match
        return jsonify({"error": "Query vector shape mismatch with document vectors"}), 500

    # Calculate similarity using cosine similarity
    sim_scores = np.dot(df.values.T, query_vector) / (np.linalg.norm(df.values.T, axis=1) * np.linalg.norm(query_vector))

    # Get top `top_n` results
    sim_sorted = np.argsort(sim_scores)[::-1][:top_n]
    results = [{"title": titles[i], "similarity_score": float(sim_scores[i]), "content": documents_clean[i]}
               for i in sim_sorted]

    return jsonify({"results": results}), 200

if __name__ == '__main__':
    app.run(debug=True)
