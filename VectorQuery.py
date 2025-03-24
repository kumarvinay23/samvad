import os
from GenerateEmbedding import VectorDBDataSource

from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer


from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")
MONGODB_ATLAS_URI= os.environ.get("MONGODB_ATLAS_URI", default=None)
MONGODB_DATABASE_NAME = os.environ.get("MONGODB_DATABASE_NAME", default=None)
MONGODB_COLLECTION_NAME = os.environ.get("MONGODB_COLLECTION_NAME", default=None)

app = Flask(__name__)

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Function to perform similarity search


@app.route('/search', methods=['POST'])
def search_document():
    try:
        data = request.get_json()
        query = data['query']
        if not query:
         return jsonify({'error': 'Missing query'}), 400
        vector_db = VectorDBDataSource(OPENAI_API_KEY, MONGODB_ATLAS_URI, MONGODB_DATABASE_NAME, MONGODB_COLLECTION_NAME).get_vector_db()
        results = vector_db.similarity_search(query)

        return jsonify([doc.page_content for doc in results]), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) #remove debug=true for production
