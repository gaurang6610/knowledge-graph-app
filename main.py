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
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_url(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    return ' '.join(p.text for p in soup.find_all('p'))

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
async def generate(request: Request, file: UploadFile = File(None), url: str = Form(None), raw_text: str = Form(None)):
    if file:
        contents = await file.read()
        text = extract_text_from_pdf(contents)
    elif url:
        text = extract_text_from_url(url)
    else:
        text = raw_text
    generate_knowledge_graph(text)
    return templates.TemplateResponse("index.html", {"request": request, "graph": True})