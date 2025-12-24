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

# ---------------- USERS (LOGIN DATA) ----------------
USERS = {
    "employee": {"password": "employee123", "role": "Employee"},
    "hr": {"password": "hr123", "role": "HR"}
}

# ---------------- CLEAN POLICY TEXT ----------------
def clean_policy_text(text):
    text = re.sub(r"\b\d+(\.\d+)+\b", "", text)
    text = re.sub(r"^\s*\d+\s*", "", text, flags=re.MULTILINE)
    return text.strip()

# ---------------- EMBEDDING ----------------
def get_embedding(text):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(res.data[0].embedding, dtype="float32")

# ---------------- LOAD RAG ----------------
def load_rag():
    loader = PyPDFLoader("Sample_HR_Policy_Document.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    texts = [c.page_content for c in chunks]
    embeddings = np.vstack([get_embedding(t) for t in texts])

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, texts

index, texts = load_rag()

# ---------------- CHAT FUNCTION ----------------
def hr_chatbot(question, history):
    q_emb = get_embedding(question)
    _, idx = index.search(np.array([q_emb]), k=3)

    context = clean_policy_text(texts[idx[0][0]])

    prompt = f"""
You are a professional HR assistant.

Rules:
- Answer only from policy
- Give a short summary (2–3 sentences)
- Do not include clause numbers
- Do not assume information

If information is missing, reply exactly:
"I checked the HR policy document, but this information is not mentioned."

Policy Content:
{context}

Question:
{question}

Final Answer:
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=120
    )

    return res.choices[0].message.content.strip()

# ---------------- LOGIN FUNCTION ----------------
def login(username, password):
    if username in USERS and USERS[username]["password"] == password:
        return True, USERS[username]["role"]
    return False, ""

# ---------------- GRADIO UI ----------------
with gr.Blocks(title="HR Policy Assistant") as demo:
    gr.Markdown("## 🔐 HR Policy Assistant Login")

    session = gr.State({"logged_in": False, "role": ""})

    with gr.Row():
        username = gr.Textbox(label="Username")
        password = gr.Textbox(label="Password", type="password")

    login_btn = gr.Button("Login")
    login_status = gr.Markdown()

    chatbot_ui = gr.ChatInterface(
        fn=hr_chatbot,
        title="🏢 HR Policy Assistant",
        visible=False
    )

    def handle_login(u, p, session_state):
        success, role = login(u, p)
        if success:
            session_state["logged_in"] = True
            session_state["role"] = role
            return (
                f"✅ Logged in as **{role}**",
                gr.update(visible=True),
                session_state
            )
        else:
            return (
                "❌ Invalid credentials",
                gr.update(visible=False),
                session_state
            )

    login_btn.click(
        handle_login,
        inputs=[username, password, session],
        outputs=[login_status, chatbot_ui, session]
    )

demo.launch()
