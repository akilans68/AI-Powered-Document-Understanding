import fitz  # PyMuPDF
import pytesseract
import cv2
import numpy as np

# Function to convert PDF to text
def convert_pdf_to_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Function to extract images from PDF
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

# Function to perform OCR on an image
def ocr_image(image_bytes, languages=None):
    if languages is None:
        languages = ['eng']
    elif isinstance(languages, str):
        languages = [languages]

    image_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    image = Image.fromarray(image_np)
    text = pytesseract.image_to_string(image, lang='+'.join(languages))
    return text

import nltk
nltk.download('punkt')
import re
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize

nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

def segment_sentences(text):
    return sent_tokenize(text)

def tokenize_sentence(sentence):
    return word_tokenize(sentence)

def preprocess_text(text):
    text = clean_text(text)
    sentences = segment_sentences(text)
    tokenized_sentences = [tokenize_sentence(sent) for sent in sentences]

    return tokenized_sentences
import os
from huggingface_hub import hf_hub_download
HUGGING_FACE_API_KEY = os.environ.get("HUGGING_FACE_API_KEY")
# LLM Integration
def load_llm_model(model_name):
    return pipeline("text-classification", model=model_name)

# Document Classification using transformers for fine-tuning or pre-trained classifier
def classify_document(text):
    # Fine-tune a transformer model
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    return outputs

# Internal Translation using Hugging Face's translation models or the translate library
def translate_text(text, target_language="de"):
    # Using Hugging Face's translation models
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-de")
    translated_chunks = []
    chunk_size = 512  # Adjust chunk size based on model limits
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        translated_chunk = translator(chunk)[0]['translation_text']
        translated_chunks.append(translated_chunk)
    return ' '.join(translated_chunks)



    # Information Extraction using transformers for NER and spaCy for entity recognition
def extract_information(text):
    # NER using transformers
    ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    entities = ner_model(text)

    # Entity recognition using spaCy
    doc = nlp(text)
    spacy_entities = [(ent.text, ent.label_) for ent in doc.ents]

    return entities, spacy_entities

# Function to process the PDF
def process_pdf(uploaded_file):
    pdf_path = uploaded_file.name
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    text = convert_pdf_to_text(pdf_path)
    images = extract_images_from_pdf(pdf_path)
    for img_bytes in images:
        img_text = ocr_image(img_bytes, languages=['eng', 'spa', 'deu'])
        text += "\n" + img_text
    return text
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
