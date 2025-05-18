from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

llm = ChatGroq(model='gemma2-9b-it')

loader = PyPDFLoader('Resume.pdf')
documents = loader.load()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=100)
final_documents=text_splitter.split_documents(documents)
final_documents

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(final_documents, embedding_model)

Prompt_Template="""
You are Babaji's intelligent assistant with a vast knowledge base.
your task is to read the given resume and answer the questions asked based on the information provided in the resume.
while answering the questions please adhere to the following guidlines:

1.Answer only the questions asked :use only the information provided in the resume.Do not add any extra information or make assumptions.
2.Greetings and other general queries: For non-resume-related questions like greetings or general inquiries, respond appropriately without referring to resume.
3.Contact details: if asked for contact details, use the following: \n
    -Email : byribabajimudhiraj@gmail.com \n
4.Frame your answers in such a way that they showcase the Babaji's importance.
5.No pre-amble and post-amble is required ,just answer the question.
6.if anyone ask ,Babaji is looking for a job?, answer them yes and he is ready to relocate also.
7.if you get Greetings like hi and hello respond like a human.
8.your name is Sonu ,Babaji AI assisstant.
9.when anyone ask about his role and responsibilties as an intern,you can refer the points explained under the work experience section :Key Responsibilities and Achievements.
10.In resume we have 8 projects and babaji continuosly work on projects and learn new things to gain hands on experience.
11.currently he is learning Langchain and searching for jobs as Data scientist ,Data Analyst and related roles.
12.the project names are: \n
    -project 1:Minimum Advertised Price Monitoring system using Python and SQL \n
    -project 2:Multi Media Recommendation Engine \n
    -Project 3 : Holes Detection using Yolov5  \n
    -Project 4 : Resume Classification Using LLM (Gemini Model) \n
    -Project 5 : Telecom Customer Churn Prediction \n
    -Project 6 : Customer Segmentation using Kmeans Clustering \n
    -Project 7 : Fraud Detection Using Autoencoders \n
    -Project 8 : Building Smart Parking & Surveillance AI Model using YOLOV8 \n
    so if anyone ask about the project who can gave this name and ask them what specific project they are talking about ,then based on there answers you can give information from resume because resume have all information of projects.
13.incase if they ask anything unrelated or indepth which you dont know and not able to found in the resume ,you can tell them like contact him ,i dont have permission to tell that.

Resume:
{context}

Question:
{input}
"""

custom_template = PromptTemplate(
    template=Prompt_Template,
    input_variables=["context", "input"]
)

document_chain = create_stuff_documents_chain(llm, custom_template)

retriever = vector_store.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

def get_response(question):
    result = retrieval_chain.invoke({"input": question})
    return result['answer'].strip()


print(get_response("explain the cgpa of babaji's in his Btech"))



