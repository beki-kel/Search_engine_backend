import pytest
import re
from io import BytesIO
from flask import Flask, jsonify, request
from app import process_text, chunk_text
import nltk
nltk.download('punkt')

# Test the process_text function (checking text cleaning and lemmatization)
def test_process_text():
    input_text = "This is a TEST text, with some punctuation!!!"
    expected_output = "this is a test text with some punctuation"  # After processing, all lowercase, punctuation removed, lemmatized

    result = process_text(input_text)
    assert result == expected_output, f"Expected: {expected_output}, but got: {result}"

# Test the chunk_text function (check if it splits text into chunks)
def test_chunk_text():
    input_text = "Sentence one. Sentence two. Sentence three. Sentence four."
    expected_output = [
        "Sentence one. Sentence two. Sentence three.",
        "Sentence four."
    ]

    result = chunk_text(input_text, sentences_per_chunk=3)
    assert result == expected_output, f"Expected: {expected_output}, but got: {result}"

# Test the /upload_pdf route (mock PDF file and query)
@pytest.fixture
def client():
    # Create a test Flask app with the necessary routes
    app = Flask(__name__)

    # Define the process_text and chunk_text functions as simple API routes for testing
    @app.route('/upload_pdf', methods=['POST'])
    def upload_pdf():
        pdf_file = request.files.get('pdf')
        query = request.form.get('query', "")
        # Mock PDF processing (you can adapt it to your real endpoint logic)
        if not pdf_file or not query:
            return jsonify({"error": "PDF and query are required"}), 400
        # Simulate returning a dummy document chunk response for testing
        return jsonify({"results": [{"title": "Document Chunk 1", "similarity_score": 0.85, "content": "This is a chunk of the document"}]}), 200

    # Create a test client
    with app.test_client() as client:
        yield client


