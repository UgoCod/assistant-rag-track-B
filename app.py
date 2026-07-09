import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

DOCS_DIR = "/tmp/documents/"

state = {}

def build_rag_pipeline():
    import urllib.request
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.prompts import ChatPromptTemplate

    os.makedirs(DOCS_DIR, exist_ok=True)
    dest = os.path.join(DOCS_DIR, "CONVENTION_SYNTEC.pdf")
    if not os.path.exists(dest):
        print("Téléchargement de CONVENTION_SYNTEC.pdf...")
        urllib.request.urlretrieve(
            "https://github.com/archiducarmel/SupDeVinci_M1_MachineLearning_DeepLearning/releases/download/datas/CONVENTION_SYNTEC.pdf",
            dest,
        )
        print("Téléchargement OK.")

    all_docs = []
    for f in sorted(os.listdir(DOCS_DIR)):
        if f.endswith(".pdf"):
            all_docs.extend(PyPDFLoader(os.path.join(DOCS_DIR, f)).load())

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100
    ).split_documents(all_docs)

    embeddings = NVIDIAEmbeddings(
        model="nvidia/llama-nemotron-embed-1b-v2", truncate="NONE"
    )
    vector_store = Chroma.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = ChatNVIDIA(
        model="openai/gpt-oss-120b", temperature=0.2, max_completion_tokens=2048
    )
    prompt = ChatPromptTemplate.from_template(
        "Tu es un assistant RH expert de la convention collective Syntec. "
        "Réponds à la QUESTION en t'appuyant UNIQUEMENT sur le CONTEXTE ci-dessous.\n"
        "Si l'information n'y figure pas, réponds exactement : « Je ne sais pas ».\n"
        "Sois concis et cite la source (document et numéro de page).\n\n"
        "CONTEXTE :\n{context}\n\nQUESTION : {input}"
    )

    def extract_answer(response):
        text = (response.content or "").strip()
        if not text:
            text = (response.additional_kwargs.get("reasoning_content", "") or "").strip()
        return text

    def format_docs(docs):
        return "\n\n".join(
            f"[{d.metadata.get('source', '?').split('/')[-1]} — page {d.metadata.get('page')}] {d.page_content}"
            for d in docs
        )

    state["retriever"] = retriever
    state["generation"] = prompt | llm | extract_answer
    state["format_docs"] = format_docs
    print("Pipeline RAG prêt.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    build_rag_pipeline()
    yield
    state.clear()


app = FastAPI(title="Assistant RH — Convention Syntec — API RAG", lifespan=lifespan)


class QuestionIn(BaseModel):
    question: str


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(payload: QuestionIn):
    if "retriever" not in state:
        raise HTTPException(status_code=503, detail="Pipeline non initialisé")
    docs = state["retriever"].invoke(payload.question)
    answer = state["generation"].invoke(
        {"context": state["format_docs"](docs), "input": payload.question}
    )
    sources = [
        {
            "document": d.metadata.get("source", "?").split("/")[-1],
            "page": d.metadata.get("page"),
        }
        for d in docs
    ]
    return {"answer": answer, "sources": sources}