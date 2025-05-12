from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
import spacy
import spacy.cli
import networkx as nx
from pyvis.network import Network
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Initialize FastAPI
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Download and load spaCy model
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_url(url):
    logging.debug(f"Fetching content from URL: {url}")
    try:
        r = requests.get(url)
        r.raise_for_status()  # Raise error for bad status codes
        soup = BeautifulSoup(r.text, "html.parser")
        text = ' '.join(p.text for p in soup.find_all('p'))
        return text
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching URL: {e}")
        return None

def generate_knowledge_graph(text):
    doc = nlp(text)
    G = nx.Graph()
    for sent in doc.sents:
        ents = [ent.text for ent in sent.ents]
        if len(ents) >= 2:
            for i in range(len(ents)-1):
                G.add_edge(ents[i], ents[i+1])
    net = Network(notebook=False)
    net.from_nx(G)
    net.save_graph("static/graph.html")

@app.get("/", response_class=HTMLResponse)
async def form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate", response_class=HTMLResponse)
async def generate(
    request: Request,
    file: UploadFile = File(None),
    url: str = Form(""),
    raw_text: str = Form("")
):
    try:
        text = ""

        logging.debug(f"Received URL: {url}")
        logging.debug(f"Received raw_text: {raw_text}")

        #  Only process the file if it's a real uploaded file
        if file and file.filename:
            contents = await file.read()
            if contents:
                text = extract_text_from_pdf(contents)
                logging.debug("PDF content processed successfully.")
            else:
                logging.debug("Uploaded file is empty, skipping.")
        
        #  If no valid PDF text, try URL
        if not text and url.strip():
            logging.debug(f"Processing URL: {url.strip()}")
            text = extract_text_from_url(url.strip())
            if text:
                logging.debug(f"Extracted text from URL: {text[:100]}...")
            else:
                return templates.TemplateResponse("index.html", {
                    "request": request,
                    "graph": False,
                    "error": "Failed to extract text from the URL."
                })

        #  If still no text, try raw_text
        if not text and raw_text.strip():
            logging.debug("Using raw text input.")
            text = raw_text.strip()
            logging.debug(f"Raw text used: {text[:100]}...")

        #  Final check
        if not text:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "graph": False,
                "error": "No valid content provided. Please upload a file, enter a URL, or paste text."
            })

        generate_knowledge_graph(text)
        return templates.TemplateResponse("index.html", {"request": request, "graph": True})

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "graph": False,
            "error": f"Error: {str(e)}"
        })
