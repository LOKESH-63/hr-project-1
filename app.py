import faiss
import gradio as gr
import numpy as np

from fastapi import FastAPI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ================= CONFIG =================
PDF_PATH = "hr_policy.pdf"
TOP_K = 8
SIMILARITY_THRESHOLD = 0.35

# ================= LOAD MODELS =================
embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")

llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=256
)

# ================= LOAD & CLEAN PDF =================
def load_pdf_text(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text.replace("\n", " ") + " "
    return text

raw_text = load_pdf_text(PDF_PATH)

# ================= CHUNKING =================
def chunk_text(text, size=750, overlap=120):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

chunks = chunk_text(raw_text)

# ================= EMBEDDINGS + FAISS =================
embeddings = embedder.encode(chunks)
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# ================= HR PROMPT =================
PROMPT = """
You are a professional HR Policy Assistant.

Answer ONLY using the policy context below.
Rewrite the answer in your own words.
Use polite, professional HR language.
Do not quote the policy text.
Do not guess.

If the answer is not found, say:
"I checked the HR policy document, but this information is not mentioned. Please contact the HR team for further clarification."

Policy Context:
{context}

Employee Question:
{question}
"""

# ================= QA FUNCTION =================
def get_answer(question):
    q_emb = embedder.encode([question])
    distances, indices = index.search(q_emb, TOP_K)

    if distances[0][0] > SIMILARITY_THRESHOLD:
        return (
            "I checked the HR policy document, but this information is not mentioned. "
            "Please contact the HR team for further clarification."
        )

    context = " ".join([chunks[i] for i in indices[0][:3]])
    prompt = PROMPT.format(context=context, question=question)

    response = llm(prompt)[0]["generated_text"]
    return response.strip()

# ================= FASTAPI =================
api = FastAPI()

@api.get("/ask")
def ask(q: str):
    return {"answer": get_answer(q)}

# ================= GRADIO UI =================
def chat(message, history):
    answer = get_answer(message)
    history.append((message, answer))
    return history, history

with gr.Blocks(title="HR Policy Assistant") as demo:
    gr.Markdown("## 🏢 HR Policy Assistant")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask your HR question...")
    state = gr.State([])

    msg.submit(chat, [msg, state], [chatbot, state])
    msg.submit(lambda: "", None, msg)

demo.launch()
