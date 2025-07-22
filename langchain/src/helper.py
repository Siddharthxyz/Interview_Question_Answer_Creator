from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import Document

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

import os
import re
from dotenv import load_dotenv
from src.prompt import PROMPT_QUESTIONS, REFINE_PROMPT_QUESTIONS



# Load Gemini API Key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY


def file_processing(file_path: str):
    # Load and read PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()
    full_text = "\n".join(page.page_content for page in data)

    # Split for question generation
    splitter_ques_gen = TokenTextSplitter(chunk_size=10000, chunk_overlap=200)
    chunks_ques_gen = splitter_ques_gen.split_text(full_text)
    document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]

    # Split again for embedding/answering
    splitter_ans_gen = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
    document_answer_gen = splitter_ans_gen.split_documents(document_ques_gen)

    return document_ques_gen, document_answer_gen


def llm_pipeline(file_path: str):
    document_ques_gen, document_answer_gen = file_processing(file_path)

    # Gemini Flash LLM for question generation
    llm_ques_gen_pipeline = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash",
        temperature=0.3,
        convert_system_message_to_human=True,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    '''# Prompt templates
    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=["text"])
    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template
    )'''

    # Question generation chain
    ques_gen_chain = load_summarize_chain(
        llm=llm_ques_gen_pipeline,
        chain_type="refine",
        verbose=True,
        question_prompt=PROMPT_QUESTIONS,
        refine_prompt=REFINE_PROMPT_QUESTIONS
    )

    ques_output = ques_gen_chain.run(document_ques_gen)

    # Clean question list
    ques_lines = ques_output.split("\n")
    filtered_ques_list = [
        line.strip()
        for line in ques_lines
        if re.match(r"^\d+\.\s", line.strip())
    ]

    # Embedding & vectorstore using Gemini embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
        )
    
    vector_store = FAISS.from_documents(document_answer_gen, embeddings)

    # Answer generation using Gemini
    llm_answer_gen = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash",
        temperature=0.1,
        convert_system_message_to_human=True,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    answer_generation_chain = RetrievalQA.from_chain_type(
        llm=llm_answer_gen,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    return answer_generation_chain, filtered_ques_list
