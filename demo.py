import os
import time
import pandas as pd
import streamlit as st
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from tqdm import tqdm
from dotenv import load_dotenv

# ==========================================
# 0. LOAD ENVIRONMENT VARIABLES
# ==========================================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_FAISS_PATH = os.getenv("DB_FAISS_PATH", "vectorstore/full_mental_health_index")
DATA_PATH = os.getenv("DATA_PATH", "Mental Health Dataset.csv")
LOG_FILE = os.getenv("LOG_FILE", "research_insight_log.csv")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 15000))

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in .env")
    st.stop()

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ==========================================
# 1. SCIENTIFIC CONFIGURATION
# ==========================================
class MentalHealthRAGSystem:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1)
        self.vector_db = self._initialize_knowledge_base()

    def _initialize_knowledge_base(self):
        if os.path.exists(DB_FAISS_PATH):
            st.sidebar.success("✅ Knowledge Base Loaded from Disk")
            return FAISS.load_local(
                DB_FAISS_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True
            )

        st.sidebar.warning("⚠️ Index not found. Starting Full Dataset Indexing...")
        return self._build_full_index()

    def _build_full_index(self):
        df = pd.read_csv(DATA_PATH)
        total_rows = len(df)
        vector_db = None

        progress_bar = st.sidebar.progress(0, text="Building Vector Space...")

        for start_idx in range(0, total_rows, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, total_rows)
            batch_df = df.iloc[start_idx:end_idx]

            docs = []
            for i, row in batch_df.iterrows():
                content = (
                    f"Profile: {row['Gender']} {row['Occupation']} in {row['Country']}. "
                    f"MH-History: {row['Mental_Health_History']}, Stress: {row['Growing_Stress']}, "
                    f"Mood: {row['Mood_Swings']}, Coping: {row['Coping_Struggles']}."
                )
                docs.append(Document(page_content=content, metadata={"id": i}))

            if vector_db is None:
                vector_db = FAISS.from_documents(docs, self.embeddings)
            else:
                vector_db.add_documents(docs)

            progress_bar.progress(end_idx / total_rows, text=f"Indexing {end_idx}/{total_rows}")

        vector_db.save_local(DB_FAISS_PATH)
        st.sidebar.success("✅ Full Indexing Complete!")
        return vector_db

    def get_qa_chain(self):
        template = """
[MHR-FRAMEWORK: CLINICAL ANALYTICS MODE]

DATA CONTEXT:
{context}

RESEARCH QUERY:
{question}

INSTRUCTIONS:
1. Summarize trends across the retrieved demographics.
2. Identify stress-to-coping correlations.
3. Report a Confidence Score based on data consistency.

ANALYTICAL INSIGHT:
"""
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_db.as_retriever(search_kwargs={"k": 15}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

# ==========================================
# 2. LOGGING
# ==========================================
def log_to_csv(query, response, latency):
    entry = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Query": query,
        "Insight": response,
        "Latency_Sec": latency
    }])

    header = not os.path.exists(LOG_FILE)
    entry.to_csv(LOG_FILE, mode="a", index=False, header=header)

# ==========================================
# 3. STREAMLIT INTERFACE
# ==========================================
def main():
    st.set_page_config(page_title="MH-RAG Framework", layout="wide")
    st.title("🧠 Retrieval-Augmented Mental Health Analytics")
    st.markdown("*A framework for data exploration and automated insight generation*")

    if "rag_system" not in st.session_state:
        st.session_state.rag_system = MentalHealthRAGSystem()

    rag = st.session_state.rag_system

    # ===== SIDEBAR BULK INPUT =====
    with st.sidebar:
        st.header("📂 Batch Analysis (Text Input)")

        bulk_text = st.text_area(
            "Enter multiple research queries (one per line)",
            height=220,
            placeholder="Analyze stress trends among students\nCompare coping strategies by gender\nIdentify mood patterns by occupation"
        )

        if st.button("🚀 Execute Bulk Processing"):
            if bulk_text.strip():
                queries = [q.strip() for q in bulk_text.split("\n") if q.strip()]
                chain = rag.get_qa_chain()

                with st.spinner(f"Processing {len(queries)} queries..."):
                    for q in queries:
                        t1 = time.time()
                        res = chain.invoke(q)
                        latency = round(time.time() - t1, 2)
                        log_to_csv(q, res["result"], latency)

                st.success(f"✅ {len(queries)} queries processed and saved to {LOG_FILE}")
            else:
                st.error("Please enter at least one query.")

    # ===== INTERACTIVE SINGLE QUERY =====
    st.subheader("Interactive Research Explorer")

    if query := st.chat_input("Enter research hypothesis..."):
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            t1 = time.time()
            chain = rag.get_qa_chain()
            response = chain.invoke(query)
            latency = round(time.time() - t1, 2)

            st.markdown(response["result"])
            st.caption(f"Performance: {latency}s | Sources Analyzed: 15")

            log_to_csv(query, response["result"], latency)

            with st.expander("View Source Evidence"):
                for doc in response["source_documents"]:
                    st.write(doc.page_content)

if __name__ == "__main__":
    os.makedirs("vectorstore", exist_ok=True)
    main()
