import gradio as gr
import faiss
import numpy as np
import os
import re
from dotenv import load_dotenv
from openai import OpenAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------- LOAD ENV ----------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------- CLEAN POLICY TEXT ----------------
def clean_policy_text(text):
    text = re.sub(r"\b\d+(\.\d+)+\b", "", text)
    text = re.sub(r"^\s*\d+\s*", "", text, flags=re.MULTILINE)
    return text.strip()

# ---------------- EMBEDDING FUNCTION ----------------
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding, dtype="float32")

# ---------------- LOAD PDF + BUILD INDEX ----------------
def load_rag_pipeline():
    loader = PyPDFLoader("Sample_HR_Policy_Document.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    texts = [c.page_content for c in chunks]
    embeddings = np.vstack([get_embedding(t) for t in texts])

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, texts

index, texts = load_rag_pipeline()

# ---------------- CHAT FUNCTION ----------------
def hr_chatbot(question, history):
    q_emb = get_embedding(question)
    _, idx = index.search(np.array([q_emb]), k=3)

    raw_context = texts[idx[0][0]]
    context = clean_policy_text(raw_context)

    prompt = f"""
You are a professional HR assistant.

Rules:
- Answer ONLY from policy content
- Give a short, natural summary (2–3 sentences)
- Do NOT include clause numbers
- Do NOT assume information

If not available, reply exactly:
"I checked the HR policy document, but this information is not mentioned."

Policy Content:
{context}

Question:
{question}

Final Answer:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=120
    )

    return response.choices[0].message.content.strip()

# ---------------- GRADIO UI ----------------
demo = gr.ChatInterface(
    fn=hr_chatbot,
    title="🏢 HR Policy Assistant",
    description="Ask questions about company HR policies",
    theme="soft"
)

demo.launch()
