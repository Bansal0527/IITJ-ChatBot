from flask import Flask, request, render_template
import os
from llama_index import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings import LangchainEmbedding
from llama_index.llms.palm import PaLM
from langchain.embeddings.gpt4all import GPT4AllEmbeddings
from llama_index import ServiceContext
from llama_index import Prompt
# os.environ['OPENAI_API_KEY'] = "your key here"
embed_model = LangchainEmbedding(GPT4AllEmbeddings())
os.environ['GOOGLE_API_KEY'] = "AIzaSyCiWI65fZsluqcnRwDxM8OXwBeu_zVypBE"
llm = PaLM()
app = Flask(__name__)
index = None





# documents = SimpleDirectoryReader('Data').load_data()
# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir='./storage')
service_context = ServiceContext.from_defaults(llm=llm, chunk_size=8000, chunk_overlap=20, embed_model=embed_model)
# index = VectorStoreIndex.from_documents(documents, service_context=service_context)
# load index
index = load_index_from_storage(storage_context,     service_context=service_context,    )

# template = (
#     "We have provided context information below. \n"
#     "---------------------\n"
#     "{context_str}"
#     "\n---------------------\n"
#     "Given this information, please answer the question and if you don't know the answer then kindly reply 'I am not sure about it.': {query_str}\n"
#     "---------------------\n"
#     "Do not guess the answer from your side. \n"
# )
# qa_template = Prompt(template)

@app.route("/")
def home():
    return render_template('index.html')


@app.route("/query", methods=["GET"])
def query_index():
  global index
  query_text = request.args.get("text", None)
  if query_text is None:
    return "No text found, please include a ?text=blah parameter in the URL", 400
  query_engine = index.as_query_engine()
  response = query_engine.query(query_text)
  return str(response), 200


if __name__ == "__main__":
    app.run(debug=True)