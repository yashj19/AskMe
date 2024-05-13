from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
import gradio as gr
import os
from dotenv import load_dotenv
load_dotenv()

DATA_PATH = "data/"
CHROMA_PATH = "chroma"
COHERE_API_KEY = cohere_api_key=os.getenv('COHERE_API_KEY')
PROMPT_TEMPLATE = """
Here are some pieces of context for the following question (they may not necessarily be sequential):

{context}

---

Answer the question based on the above context: {query}
"""
# embedding_func = CohereEmbeddings(COHERE_API_KEY)
embedding_func = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))

def load_documents(file):
  loader = DirectoryLoader(DATA_PATH, glob="*.md")
  if not file is None:
    loader = PyPDFLoader(file)
  documents = loader.load()
  return documents

def split_into_chunks(documents):
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=500,
    length_function=len,
    add_start_index=True
  )
  
  # contain source file name and start index in metadata (really useful)
  chunks = text_splitter.split_documents(documents)
  return chunks

def save_to_chroma(chunks: list[Document]):
  Chroma.from_documents(chunks, embedding_func, persist_directory=CHROMA_PATH)
  
def generate_db(file=None):
  docs = load_documents(file)
  chunks = split_into_chunks(docs)
  save_to_chroma(chunks)
  print("Finished creating chroma DB")

def process_query(query_text, file):

  if not file is None:
    generate_db(file)

  db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_func)

  # return type of the search List[Tuple[Document, float]]
  results = db.similarity_search_with_relevance_scores(query_text, k=7)

  # make sure we find relevant info before moving to next step of the process
  if len(results) == 0 or results[0][1] < 0.7:
    print("unable to find matching results")
    return "unable to find matching results", "no response"
  
  output = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
  prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
  prompt = prompt_template.format(context=output, query=query_text)

  # set up cohere model
  model = ChatCohere(model="command-r-plus", cohere_api_key=COHERE_API_KEY)
  response = model.invoke(prompt).content

  return response, prompt

def take_console_input():
  # query repeatedly
  while True:
    query_text = input("Enter your query: ")

    if query_text == "done":
      break

    # process
    response, prompt = process_query(query_text)
    print("Prompt: ", prompt)
    print("Response: ", response)

def run_gradio_app():
  demo = gr.Interface(
    process_query,
    inputs=["text", gr.File(type="filepath", label="Upload a PDF")],
    outputs=["text", "text"],
    title="Document QA Chatbot",
    description="Give a pdf document to ask questions about. If not given, will have the context of the 'Alice In Wonderland' book."
  )

  demo.launch()

def main():
  if not os.path.exists(CHROMA_PATH):
    generate_db()
  run_gradio_app()
  # take_console_input()
  pass

if __name__ == "__main__":
  main()