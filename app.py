import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from dotenv import load_dotenv

import os
import uuid


# ---------------- LOAD ENVIRONMENT VARIABLES ----------------

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Resume & Job Description Analyzer",
    page_icon="📄",
    layout="wide"
)


# ---------------- FUNCTIONS ----------------

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    return text_splitter.split_text(text)


def extract_pdf_text(pdf_file):
    text = ""

    reader = PdfReader(pdf_file)

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_groq_model():
    return ChatGroq(
        groq_api_key=groq_api_key,
        model="openai/gpt-oss-20b",
        temperature=0.2
    )


embedding_model = load_embedding_model()
llm = load_groq_model()


# ---------------- TITLE ----------------

st.title("📄 AI Resume & Job Description Analyzer")

st.write(
    "Upload your resume and paste a job description to analyze "
    "skills, gaps, suitability, and interview preparation using RAG."
)


# ---------------- INPUT SECTION ----------------

col1, col2 = st.columns(2)


with col1:

    st.subheader("Upload Resume")

    resume_file = st.file_uploader(
        "Upload your resume PDF",
        type=["pdf"]
    )


with col2:

    st.subheader("Job Description")

    jd_text = st.text_area(
        "Paste the job description",
        height=250
    )


# ---------------- PROCESS DATA ----------------

if resume_file is not None and jd_text:

    resume_text = extract_pdf_text(
        resume_file
    )

    resume_chunks = split_text(
        resume_text
    )

    jd_chunks = split_text(
        jd_text
    )


    all_chunks = (
        resume_chunks
        +
        jd_chunks
    )


    metadata = []


    for chunk in resume_chunks:

        metadata.append(
            {
                "source": "resume"
            }
        )


    for chunk in jd_chunks:

        metadata.append(
            {
                "source": "job_description"
            }
        )


    collection_name = (
        f"resume_job_{uuid.uuid4().hex}"
    )


    vectorstore = Chroma.from_texts(
        texts=all_chunks,
        embedding=embedding_model,
        metadatas=metadata,
        collection_name=collection_name
    )


    st.success(
        "Resume and Job Description processed successfully!"
    )


    # ---------------- ANALYSIS OPTIONS ----------------

    st.subheader("Choose Analysis")

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        match_button = st.button(
            "Match Skills"
        )


    with col2:

        missing_button = st.button(
            "Missing Skills"
        )


    with col3:

        suitability_button = st.button(
            "Overall Suitability"
        )


    with col4:

        interview_button = st.button(
            "Interview Questions"
        )


    # ---------------- CUSTOM QUESTION ----------------

    st.subheader(
        "Ask Your Own Question"
    )

    custom_question = st.text_input(
        "Ask anything about the resume and job description"
    )


    analyze_custom = st.button(
        "Analyze Question"
    )


    # ---------------- SELECT QUESTION ----------------

    question = None


    if match_button:

        question = (
            "What skills in the resume match "
            "the job description?"
        )


    elif missing_button:

        question = (
            "What important skills required by the "
            "job description are missing or weak "
            "in the resume?"
        )


    elif suitability_button:

        question = (
            "Evaluate the candidate's overall suitability "
            "for this job based on the resume and "
            "job description."
        )


    elif interview_button:

        question = (
            "Generate relevant technical and HR interview "
            "questions for this candidate based on "
            "the resume and job description."
        )


    elif analyze_custom and custom_question:

        question = custom_question


    # ---------------- RAG RETRIEVAL ----------------

    if question:

        with st.spinner(
            "Analyzing resume and job description..."
        ):

            resume_results = (
                vectorstore.similarity_search(
                    question,
                    k=4,
                    filter={
                        "source": "resume"
                    }
                )
            )


            jd_results = (
                vectorstore.similarity_search(
                    question,
                    k=4,
                    filter={
                        "source": "job_description"
                    }
                )
            )


            resume_context = "\n\n".join(
                [
                    doc.page_content
                    for doc in resume_results
                ]
            )


            jd_context = "\n\n".join(
                [
                    doc.page_content
                    for doc in jd_results
                ]
            )


            # ---------------- RAG PROMPT ----------------

            prompt = f"""
You are an AI Resume and Job Description Analyzer.

Answer the user's question using only the evidence
provided in the Resume Context and Job Description Context.

Do not invent skills, experience, education,
projects, achievements, or requirements.

If something is not available in the provided context,
say that it is not clearly found.

User Question:
{question}

Resume Context:
{resume_context}

Job Description Context:
{jd_context}

Give a clear and professional answer.

When relevant, organize the response using:

- Matching Skills
- Missing or Weak Skills
- Relevant Projects or Experience
- Overall Suitability
- Recommended Improvements

For interview-question requests, provide both
technical and HR questions.

Keep the answer evidence-based and easy to read.
"""


            try:

                response = llm.invoke(
                    prompt
                )


                st.subheader(
                    "🤖 AI Analysis"
                )


                st.markdown(
                    response.content
                )


            except Exception as error:

                st.error(
                    f"Groq API Error: {error}"
                )


    # ---------------- OPTIONAL DETAILS ----------------

    with st.expander(
        "View Retrieved RAG Context"
    ):

        if question:

            st.write(
                "### Resume Context"
            )

            for i, doc in enumerate(
                resume_results
            ):

                st.write(
                    f"Resume Chunk {i + 1}"
                )

                st.write(
                    doc.page_content
                )


            st.write(
                "### Job Description Context"
            )

            for i, doc in enumerate(
                jd_results
            ):

                st.write(
                    f"JD Chunk {i + 1}"
                )

                st.write(
                    doc.page_content
                )


else:

    st.info(
        "Upload a resume PDF and paste a job description to begin."
    )