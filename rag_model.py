import os
import re
import pickle
import numpy as np
from typing import List, Dict
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import zipfile
import tempfile
import shutil
import pymupdf
from pathlib import Path

class RAGModel:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        load_dotenv()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        if not self.gemini_api_key:
            raise ValueError("Google Gemini API key not found in the .env file.")
        
        self.model_name = model_name
        self.model = self._initialize_model()
        self.embeddings = None
        self.chunks = None

    def _initialize_model(self):
        try:
            model = SentenceTransformer(self.model_name)
            return model
        except Exception as e:
            raise ValueError(f"Model initialization error: {e}")

    def preprocess_text(self, file_path: str, dest_file_path: str, perc_thresh=0.35, num_space_fr=0.47):
        try:
            with open(file_path, 'r', encoding="utf8") as f:
                text = f.read()

            lines = text.split('\n')
            line_len_thresh = self._thresh_decider(file_path, perc_thresh)

            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if not line or len(line) <= line_len_thresh:
                    continue
                
                line = re.sub(r'[^\w\s.]', '', line)
                line = re.sub(r'\s+', ' ', line)
                line = line.strip()

                if ''.join(line.split()).isdigit() or ('Fig' in line) or ('Table' in line):
                    continue

                char_count = sum(1 for char in line if char.isalpha() and char != ' ')
                num_count = sum(1 for char in line if char.isdigit())
                sp_count = sum(1 for char in line if char.isspace())

                if len(line) == 0 or (num_count + sp_count) / len(line) >= num_space_fr:
                    continue

                cleaned_lines.append(line)

            with open(dest_file_path, 'w', encoding="utf8") as f:
                f.write('\n'.join(cleaned_lines))

            return dest_file_path
        except Exception as e:
            raise ValueError(f"Error in preprocessing text: {e}")

    def _thresh_decider(self, file_path: str, perc_thresh=0.35):
        with open(file_path, 'r', encoding="utf8") as f:
            text = f.read()
        lines = text.split('\n')
        line_len_arr = [len(line) for line in lines]
        return min(line_len_arr) + (max(line_len_arr) - min(line_len_arr)) * perc_thresh

    def chunk_text(self, text: str, chunk_size: int = 1000) -> List[Dict]:
        words = text.split()
        chunks = []
        current_chunk = []

        for word in words:
            if sum(len(w) + 1 for w in current_chunk) + len(word) <= chunk_size:
                current_chunk.append(word)
            else:
                chunks.append({"chunk_text": " ".join(current_chunk)})
                current_chunk = [word]

        if current_chunk:
            chunks.append({"chunk_text": " ".join(current_chunk)})

        return chunks

    def process_file(self, file_path: str, chunk_size: int = 1000):
        try:
            with open(file_path, 'r', encoding="utf8") as file:
                content = file.read()
            self.chunks = self.chunk_text(content, chunk_size)
            return self.chunks
        except Exception as e:
            raise ValueError(f"Error processing file: {e}")

    def generate_embeddings(self):
        if not self.chunks:
            raise ValueError("No chunks available. Process a file first.")

        chunk_texts = [chunk['chunk_text'] for chunk in self.chunks]
        self.embeddings = self.model.encode(chunk_texts, convert_to_tensor=True, show_progress_bar=True)

    def save_model_and_embeddings(self, model_path='model.pkl', embeddings_path='embeddings.pkl'):
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(embeddings_path, 'wb') as f:
            pickle.dump(self.embeddings, f)

    def load_model_and_embeddings(self, model_path='model.pkl', embeddings_path='embeddings.pkl'):
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(embeddings_path, 'rb') as f:
                self.embeddings = pickle.load(f)
        except Exception as e:
            raise ValueError(f"Error loading model or embeddings: {e}")

    def query(self, query: str, top_k: int = 3):
        if self.embeddings is None or self.chunks is None:
            raise ValueError("Embeddings or chunks not available. Process a file and generate embeddings first.")

        query_embedding = self.model.encode(query, convert_to_numpy=True).reshape(1, -1)
        embeddings_array = self.embeddings.cpu().numpy()

        similarities = np.dot(embeddings_array, query_embedding.T).flatten() / (
            np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(query_embedding)
        )

        top_indices = np.argsort(similarities)[-min(top_k, len(similarities)):][::-1]
        top_chunks = [self.chunks[idx]['chunk_text'] for idx in top_indices if idx < len(self.chunks)]

        return self._generate_response(query, top_chunks)

    def _generate_response(self, query: str, context_chunks: List[str]):
        context = ' '.join(context_chunks)
        max_context_length = 2048
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."

        input_text = f"Query: {query}\nContext: {context}"

        try:
            genai.configure(api_key=self.gemini_api_key)
            gen_model = genai.GenerativeModel("gemini-1.5-flash")
            response = gen_model.generate_content(input_text)
            
            if not hasattr(response, 'text'):
                raise ValueError("Invalid response format from Gemini API")
            
            return response.text
        except Exception as e:
            raise ValueError(f"Error generating response from Gemini API: {e}")

def move_pdfs(source_folder: str, destination_folder: str) -> None:
    os.makedirs(destination_folder, exist_ok=True)
    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.lower().endswith(".pdf"):
                source_path = os.path.join(root, file)
                try:
                    os.makedirs(destination_folder, exist_ok=True)
                    shutil.copy2(source_path, os.path.join(destination_folder, file))
                except Exception as e:
                    print(f"Error copying {file}: {str(e)}")
    print("All PDFs have been moved to the new folder.")

def extract_text_from_pdfs(folder_path: str, output_file: str, src_folder: str) -> None:
    all_text = []
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        for filename in os.listdir(folder_path):
            if not filename.lower().endswith('.pdf'):
                continue
            pdf_path = os.path.join(folder_path, filename)
            print(f"Processing: {filename}")
            try:
                doc = pymupdf.open(pdf_path)
                rel_path = os.path.relpath(pdf_path, src_folder)
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text("text")
                    if text.strip():
                        metadata = f"\n\n\n{rel_path}|{page_num + 1}\n"
                        all_text.extend([metadata, text])
                doc.close()
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
        if all_text:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(all_text))
            print(f"\nText extraction completed. Output saved to: {output_file}")
        else:
            print("No text was extracted from the PDFs.")
    except Exception as e:
        print(f"An error occurred during text extraction: {str(e)}")
        raise

def pipeline_function(zip_file, rag_model):
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            with zipfile.ZipFile(zip_file.name, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            pdf_folder = os.path.join(temp_dir, "pdfs")
            os.makedirs(pdf_folder, exist_ok=True)
            move_pdfs(temp_dir, pdf_folder)
            
            extracted_text_file = os.path.join(temp_dir, "extracted_text.txt")
            extract_text_from_pdfs(pdf_folder, extracted_text_file, temp_dir)
            
            processed_file = os.path.join(temp_dir, "processed_text.txt")
            rag_model.preprocess_text(extracted_text_file, processed_file)
            
            rag_model.chunks = rag_model.process_file(processed_file)
            rag_model.generate_embeddings()
            rag_model.save_model_and_embeddings()
            
            return "Pipeline completed successfully. Model and embeddings are ready for use."
        except Exception as e:
            return f"Error in pipeline execution: {str(e)}"
