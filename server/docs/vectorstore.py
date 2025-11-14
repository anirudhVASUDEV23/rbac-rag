import os
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone,ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
import voyageai
import asyncio


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

UPLOAD_DIR="./uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

pc=Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
spec=ServerlessSpec(cloud="aws",region=PINECONE_ENV)
existing_index=[i["name"] for i in pc.list_indexes()]

if PINECONE_INDEX_NAME not in existing_index:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1024, 
        metric="dotproduct",
        spec=spec
        )
    while not pc.describe_index(PINECONE_INDEX_NAME).status.ready:
        print("Index is being created, waiting for 1 second...")
        time.sleep(1)

index=pc.Index(PINECONE_INDEX_NAME)

# -------------------VoyageAI Embeddings Class -------------------
class VoyageAIEmbeddings(Embeddings):
    def __init__(self, model_name="voyage-3.5-lite", device="cpu"):
        self.client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
        self.model_name = model_name
        self.device = device

    def embed_documents(self, texts):
        embeddings_list = []
        max_batch = 8  # Voyage AI max batch per request
        for i in range(0, len(texts), max_batch):
            batch_texts = texts[i:i + max_batch]
            response = self.client.embed(
                batch_texts,
                model=self.model_name,
                input_type="document"
            )
            embeddings_list.extend(response.embeddings)
        return embeddings_list

    def embed_query(self, query):
        response = self.client.embed(
            [query],
            model=self.model_name,
            input_type="query"
        )
        return response.embeddings[0]


embed_model = VoyageAIEmbeddings(model_name="voyage-3.5-lite", device="cpu")


async def load_vectorstore(uploaded_files,role:str,doc_id:str):
    embed_model = VoyageAIEmbeddings(model_name="voyage-3.5-lite", device="cpu")

    for file in uploaded_files:
        save_path=Path(UPLOAD_DIR).joinpath(file.filename)
        with open(save_path,"wb") as f:
            f.write(file.file.read())
        
        
        loader=PyPDFLoader(str(save_path))
        documents=loader.load()
        print("Role for document {}: {}".format(file.filename,role))

        splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
        chunks=splitter.split_documents(documents)

        texts=[chunk.page_content for chunk in chunks]
        ids=[f"{doc_id}-{i}" for i in range(len(chunks))]
        metadata=[
            {
                "source":file.filename,
                "role":role,
                "doc_id":doc_id,
                "page":chunk.metadata.get("page",0),
                ## Storing the text content as metadata for reference to LLMs
                "text": chunk.page_content
            }
            for i,chunk in enumerate(chunks)
        ]

        print(f"Embedding {len(texts)} chunk...")
        embeddings=await asyncio.to_thread(embed_model.embed_documents,texts)
        

        ##uploading to the pinecone index
        index.upsert(vectors=list(zip(ids,embeddings,metadata)))
        with tqdm(total=len(embeddings),desc="Uploading to Pinecone") as pbar:
            index.upsert(vectors=zip(ids,embeddings,metadata))
            pbar.update(len(embeddings))
        print("Upload complete for {}.".format(file.filename))



# 🤝 How upsert works with your code

# Let's say:

# ids = ["id1", "id2"]
# embeddings = [[0.1, 0.2, 0.3], [0.5, 0.6, 0.7]]
# metadata = [{"role": "patient"}, {"role": "doctor"}]


# zip(ids, embeddings, metadata) produces:

# [
#   ("id1", [0.1, 0.2, 0.3], {"role": "patient"}),
#   ("id2", [0.5, 0.6, 0.7], {"role": "doctor"})
# ]


# This is sent to Pinecone, and Pinecone stores 2 vectors.

# 🔍 Important Clarification
# ❌ score is not stored
# ❌ Upsert does not take or store a score
# ✔️ Score only appears when you do .query()
# ✔️ Score depends on the query embedding


