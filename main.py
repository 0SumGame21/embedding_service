from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np
import os

app = Flask(__name__)

# Load model once at startupssss
print("Loading embedding model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("Model ready!")

@app.route('/embed', methods=['POST'])
def embed_text():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
            
        # Generate embedding
        embedding = model.encode(text)
        
        # Normalize for cosine similarity
        normalized = embedding / np.linalg.norm(embedding)
        
        return jsonify({
            'embedding': normalized.tolist(),
            'dimensions': len(normalized)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)