from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Building Machine Learning Systems with Python - Second Edition copy.pdf")

docs = loader.load()

spilitter = CharacterTextSplitter(
    separator="",
    chunk_size=100,
    chunk_overlap=0,
)

texts = spilitter.split_documents(docs)
print(texts[10])