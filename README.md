## **AI-Powered-Document-Understanding and Processing Pipeline**

Welcome to the AI-Powered-Document-Understanding and Processing Pipeline! This project is designed to streamline the process of extracting text, images, and information from PDF documents. Additionally, it offers features for text classification, translation, and entity recognition.

# **Features**
PDF to Text Conversion: Extracts and converts the text from PDF documents.
Image Extraction and OCR: Extracts images from PDFs and performs OCR (Optical Character Recognition) to convert images into text.
Text Preprocessing: Cleans and tokenizes text for further analysis.
Information Extraction: Uses transformers and spaCy for Named Entity Recognition (NER).
Document Classification: Classifies documents using pre-trained transformer models.
Text Translation: Translates text to a specified language using Hugging Face's translation models.

# **Installation**

# Install the required packages:

#!apt-get update

#!apt-get install -y tesseract-ocr

#!apt-get install -y libtesseract-dev

#!pip install fitz

#!pip install pytesseract

#pip install pymupdf

#pip install pytesseract

#!pip install spacy

#!pip install transformers

#!pip install pdf2image

#!pip install Pillow

#pip install Translate

#!pip install streamlit -q


# Download the necessary NLTK data:

import nltk

nltk.download('punkt')

# Set up your Hugging Face API key:

export HUGGING_FACE_API_KEY=your_api_key

in the huggingface.co

# **Components**
# PDF to Text Conversion
# This function converts the content of a PDF file to text.

def convert_pdf_to_text(pdf_path):

    doc = fitz.open(pdf_path)
    
    text = ""
    
    for page_num in range(len(doc)):
        
        page = doc.load_page(page_num)
        
        text += page.get_text()
    
    return text

# Image Extraction and OCR
# Extracts images from PDF and performs OCR to convert them into text.

def extract_images_from_pdf(pdf_path):
    
    doc = fitz.open(pdf_path)
    
    images = []
    
    for page_num in range(len(doc)):
        
        page = doc.load_page(page_num)
        
        for img in page.get_images(full=True):
            
            xref = img[0]
            
            base_image = doc.extract_image(xref)
            
            image_bytes = base_image["image"]
            
            images.append(image_bytes)
    
    return images
    
# Text Preprocessing
# Cleans and tokenizes text for further analysis.

def preprocess_text(text):
    
    text = clean_text(text)
    
    sentences = segment_sentences(text)
    
    tokenized_sentences = [tokenize_sentence(sent) for sent in sentences]
    
    return tokenized_sentences
    
# Information Extraction
# Uses transformers and spaCy for Named Entity Recognition (NER).

def extract_information(text):
    
    ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    
    entities = ner_model(text)

    doc = nlp(text)
    
    spacy_entities = [(ent.text, ent.label_) for ent in doc.ents]

    return entities, spacy_entities

# Document Classification
# Classifies documents using a pre-trained transformer model.

def classify_document(text):
    
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    inputs = tokenizer(text, return_tensors="pt")
    
    outputs = model(**inputs)
    
    return outputs
    
# Text Translation
# Translates text to a specified language using Hugging Face's translation models.

def translate_text(text, target_language="de"):
    
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-de")
    
    translated_chunks = []
    
    chunk_size = 512
    
    for i in range(0, len(text), chunk_size):
    
        chunk = text[i:i+chunk_size]
        
        translated_chunk = translator(chunk)[0]['translation_text']
        
        translated_chunks.append(translated_chunk)
    
    return ' '.join(translated_chunks)

# Streamlit App
# Provides a user-friendly interface for uploading PDF files and performing various document processing tasks.

import streamlit as st

st.title("Document Processing Chatbot")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    
    with st.spinner("Processing..."):
        
        text = process_pdf(uploaded_file)
        
        preprocessed_text = preprocess_text(text)
        
        st.success("File processed successfully!")
        
        st.text_area("Extracted Text", text, height=300)

    
    if st.button("Extract Information"):
        
        with st.spinner("Extracting information..."):
            
            entities, spacy_entities = extract_information(text)
            
            st.write("Entities (NER):", entities)
            
            st.write("Entities (spaCy):", spacy_entities)

    
    if st.button("Classify Document"):
        
        with st.spinner("Classifying document..."):
            
            classification = classify_document(text)
            
            st.json(classification)

    
    if st.button("Translate Text"):
        
        with st.spinner("Translating text..."):
            
            translation = translate_text(text)
            
            st.text_area("Translated Text", translation, height=300)

# Models use in the piplines

extract_information : dbmdz/bert-large-cased-finetuned-conll03-english

classify_document : distilbert-base-uncased-finetuned-sst-2-english

translate_text : Helsinki-NLP/opus-mt-en-de

# Run the streamlit

# !wget -q -O - ipv4.icanhazip.com
(it create a specific tunnel password)

create an app.py and past the full code and save it 

# !streamlit run app.py & npx localtunnel --port 8501
use this cmd to run the streamlit
