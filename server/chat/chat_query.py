import os
import asyncio
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import voyageai
from langchain.embeddings.base import Embeddings

load_dotenv()

PINCECONE_API_KEY=os.getenv("PINECONE_API_KEY")
PINECONE_ENV=os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME=os.getenv("PINECONE_INDEX_NAME")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

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
    

pc=Pinecone(api_key=PINCECONE_API_KEY, environment=PINECONE_ENV)
index=pc.Index(PINECONE_INDEX_NAME)
embed_model = VoyageAIEmbeddings(model_name="voyage-3.5-lite", device="cpu")
llm=ChatGroq(
    temperature=0.3,
    model_name="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY,
    )

prompt=PromptTemplate.from_template(
    """
    You are a helpful healthcare assistant,Answer the following question based only on the provided context.

    Question:{question}

    Context:{context}
    If the context if empty or \n or multiple \n just say that the user they dont have the required permission to access the answer to their questions in a polite answer and say that the admin hasn't posted any documents related to the user's query and ask them to contact the admin for further clarification and end the conversation poiltely and in this case dont provide the document source in your answer
    Only Include the document source if relevant in your answer and when the context is not empty or \n or multiple \n
    """
)


rag_chain=prompt|llm

async def answer_query(query:str,user_role:str):
    embedding=await asyncio.to_thread(embed_model.embed_query,query)
    results=await asyncio.to_thread(index.query,vector=embedding,top_k=3,include_metadata=True)

    filtered_contexts=[]
    sources=set()

    for match in results["matches"]:
        metadata=match["metadata"]
        if metadata.get("role")==user_role:
            print(metadata.get("text",""))
            filtered_contexts.append(metadata.get("text","")+"\\n")
            sources.add(metadata.get("source"))
        
    if not filtered_contexts:
        return {"answer":"No relevant info found"}
    
    docs_text="\\n".join(filtered_contexts)
    final_answer=await asyncio.to_thread(rag_chain.invoke,{"question":query,"context":docs_text})
    # print("Context is:",docs_text)
    # print(final_answer.content)

    return {
        "answer":final_answer.content,
        "sources":list(sources)
    }