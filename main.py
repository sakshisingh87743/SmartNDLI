from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import numpy as np

app = FastAPI(title="SmartNDLI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

items = [
    {"title": "Introduction to Algorithms", "author": "Cormen, Leiserson, Rivest, Stein", "desc": "Algorithms, sorting, graphs, dynamic programming ki standard book"},
    {"title": "Let Us C", "author": "Yashavant Kanetkar", "desc": "C programming सीखने की शुरुआती किताब, beginners के लिए"},
    {"title": "Python Crash Course", "author": "Eric Matthes", "desc": "Hands-on introduction to Python programming with projects"},
    {"title": "Artificial Intelligence: A Modern Approach", "author": "Stuart Russell, Peter Norvig", "desc": "AI ki foundation book: search, logic, learning, intelligent agents"},
    {"title": "Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow", "author": "Aurelien Geron", "desc": "Practical machine learning aur neural networks code ke saath"},
    {"title": "Deep Learning", "author": "Ian Goodfellow, Yoshua Bengio, Aaron Courville", "desc": "Neural networks aur deep learning ka theory"},
    {"title": "Clean Code", "author": "Robert C. Martin", "desc": "Achha, readable aur maintainable code likhne ke tareeke"},
    {"title": "Operating System Concepts", "author": "Silberschatz, Galvin, Gagne", "desc": "Process, memory management, scheduling, file systems"},
    {"title": "Database System Concepts", "author": "Silberschatz, Korth, Sudarshan", "desc": "डेटाबेस, SQL, normalization और transactions की जानकारी"},
    {"title": "Computer Networking: A Top-Down Approach", "author": "Kurose, Ross", "desc": "Internet, TCP/IP, protocols aur networks kaise kaam karte hain"},
]

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
vecs = model.encode([f"{i['title']} {i['author']} {i['desc']}" for i in items], normalize_embeddings=True)

@app.get("/search")
def search(q: str, limit: int = 5):
    qv = model.encode([q], normalize_embeddings=True)[0]
    scores = vecs @ qv
    top = np.argsort(scores)[::-1][:limit]
    return [{**items[i], "score": round(float(scores[i]), 3)} for i in top]