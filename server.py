
from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
import io
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
from sentence_transformers import CrossEncoder # Import CrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = Flask(__name__)
CORS(app)

embedding_model = None
qa_pipeline = None

try:
    # Initialize models
    # Sentence Transformer for embeddings
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    # Question Answering pipeline
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2") # Upgraded model

    # Cross-Encoder for re-ranking
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

except Exception as e:
    print(f"Error loading AI models: {e}")
    # Handle case where models fail to load, e.g., exit or set a flag

# Global variables to store PDF text and FAISS index
app.config['pdf_text_chunks'] = []
app.config['faiss_index'] = None
app.config['pdf_texts'] = []

def chunk_text(text, chunk_size=1000, overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    if embedding_model is None or qa_pipeline is None:
        return jsonify({'error': 'AI models failed to load. Please check server logs.'}), 500

    if 'pdf' not in request.files:
        return jsonify({'error': 'No PDF file provided'}), 400

    file = request.files['pdf']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.endswith('.pdf'):
        try:
            reader = PdfReader(io.BytesIO(file.read()))
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() + "\n"
            
            # Chunk the extracted text
            chunks = chunk_text(full_text)
            app.config['pdf_text_chunks'] = chunks
            
            # Generate embeddings for chunks
            chunk_embeddings = embedding_model.encode(chunks)
            
            # Create FAISS index
            dimension = chunk_embeddings.shape[1]
            faiss_index = faiss.IndexFlatL2(dimension)
            faiss_index.add(np.array(chunk_embeddings).astype('float32'))
            app.config['faiss_index'] = faiss_index
            app.config['pdf_texts'] = chunks # Store chunks for retrieval

            return jsonify({'message': 'PDF uploaded and processed successfully'}), 200
        except Exception as e:
            print(f"Error during PDF processing: {e}") # Added for explicit logging
            import traceback
            return jsonify({'error': f"PDF processing failed: {str(e)}\n{traceback.format_exc()}"}), 500
    else:
        return jsonify({'error': 'Invalid file type. Only PDF is supported.'}), 400

@app.route('/ask-pdf', methods=['POST'])
def ask_pdf():
    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    faiss_index = app.config.get('faiss_index')
    pdf_texts = app.config.get('pdf_texts')

    if not faiss_index or not pdf_texts:
        return jsonify({'error': 'No PDF uploaded yet. Please upload a PDF first.'}), 400

    try:
        # Generate embedding for the question
        question_embedding = embedding_model.encode([question])
        
        # Search FAISS index for an initial set of relevant chunks (e.g., k=10-15)
        initial_k = 10 # Retrieve more chunks initially for re-ranking
        D, I = faiss_index.search(np.array(question_embedding).astype('float32'), k=initial_k)
        
        initial_relevant_chunks = [pdf_texts[idx] for idx in I[0]]

        # Pair question with each initial relevant chunk for re-ranking
        pairs = [[question, chunk] for chunk in initial_relevant_chunks]
        # Get relevance scores from the cross-encoder
        rerank_scores = reranker.predict(pairs)

        # Sort chunks by re-rank scores and select the top ones
        top_k_reranked = 5 # Select top 5 after re-ranking
        ranked_chunks_indices = np.argsort(rerank_scores)[::-1]
        top_relevant_texts = [initial_relevant_chunks[idx] for idx in ranked_chunks_indices[:top_k_reranked]]
        
        combined_relevant_text = "\n\n".join(top_relevant_texts)

        # Enhanced prompt engineering for the QA pipeline
        # By clearly demarcating context and adding strict instructions, we guide the model
        formatted_context = f"Context: {combined_relevant_text}\n\nBased *only* on the provided context, answer the following question. If the information is not present in the context, state that the answer cannot be found in the document."
        
        # Use the QA pipeline to get the answer
        q_input = {
            'question': question,
            'context': formatted_context
        }
        answer = qa_pipeline(q_input)['answer']

        return jsonify({'answer': answer}), 200
    except Exception as e:
        print('Error answering question:', e)
        import traceback
        return jsonify({'error': f"An error occurred while getting an answer: {str(e)}\n{traceback.format_exc()}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
