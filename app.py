import streamlit as st
import os
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from PyPDF2 import PdfReader
import textwrap

# Streamlit Sidebar
with st.sidebar:
    st.title('PDF Answering AI')
    st.markdown('''
    ## About
    This app is made using 'bert-large-uncased-whole-word-masking-finetuned-squad' 
    model from Hugging Face's Transformers library, known for
    its robustness and accuracy in question-answering tasks

    ''')
    st.write('Made by [Amey Somvanshi](https://www.linkedin.com/in/amey-somvanshi-66395225b)')

# Main Function
def main():
    st.header("PDF ANSWERING AI")

    # Upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()



        model_path = os.path.join('pretrained', 'model')
        tokenizer_path = os.path.join('pretrained', 'tokenizer')

        # Check if the model and tokenizer directories exist
        if not os.path.exists(model_path):
            st.error(f"Model directory not found: {model_path}")
            return

        if not os.path.exists(tokenizer_path):
            st.error(f"Tokenizer directory not found: {tokenizer_path}")
            return

        # Load the model and tokenizer
        model = AutoModelForQuestionAnswering.from_pretrained(model_path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

        query = st.text_input("Ask questions about your PDF file:")

        if query:
            # Split the text into overlapping chunks to handle long contexts
            def create_chunks(text, max_chunk_length, overlap):
                chunks = []
                start = 0
                while start < len(text):
                    end = min(start + max_chunk_length, len(text))
                    chunks.append(text[start:end])
                    start += max_chunk_length - overlap
                return chunks


            max_chunk_length = 500
            overlap = 100  # Overlap between chunks to catch split information

            chunks = create_chunks(text, max_chunk_length, overlap)

            best_answer = ""
            highest_score = 0

            for chunk in chunks:
                input_data = {
                    'question': query,
                    'context': chunk
                }
                res = nlp(input_data)
                if res['score'] > highest_score:
                    highest_score = res['score']
                    best_answer = res['answer']

            st.write(f"Answer: {best_answer}")
            st.write(f"Confidence: {highest_score*100:.2f}%")

if __name__ == '__main__':
    main()


