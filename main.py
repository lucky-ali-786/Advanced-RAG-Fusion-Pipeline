from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from google import genai
import json
from google.genai import types
file_path = Path(__file__).parent / "Interupts-1.pdf"
loader = PyPDFLoader(file_path=file_path)
docs=loader.load()
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",api_key="")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs=text_splitter.split_documents(documents=docs)
retreiver=QdrantVectorStore.from_existing_collection(
        url="http://localhost:6333",
        collection_name="my_first_vector_db",
        embedding=embeddings
)
result_final=[]
SYSTEM_PROMPT="""
YOU ARE AN AI AGENT WHICH WRITES THE USER'S QUERY IN THREE DIFFERENT AND RETURNS the OUTPUTS SHOWN BELOW
Example->
User's Query->what is instruction cycle ?
Output-> "what do you mean by instruction cycle ?","explain the life cycle of instruction ?","what is cycle of decode,fetch,execute ?"
"""
client = genai.Client(api_key="")
content=client.models.generate_content(
     model="gemini-2.5-flash-lite",
     contents="EXPLAIN THE INSTRUCTION CYCLE?",
    config=types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.2,
    ),
)
myarr=content.text.split(',')
for i in range(0,3):
    result_query=retreiver.similarity_search(
        query=myarr[i]
    )
    chunks = [f"Page: {i.metadata.get('page', 'Unknown')} \n{i.page_content}" for i in result_query]
    result_final.append(chunks)
# print(result_final)
def reciprocal_rank_fusion(list_of_ranked_lists, k=60):
    rrf_scores = {}
    for current_list in list_of_ranked_lists:
        rank = 1 
        for chunk_string in current_list:
            if chunk_string not in rrf_scores:
                rrf_scores[chunk_string] = 0.0
            math_score = 1.0 / (k + rank)
            rrf_scores[chunk_string] = rrf_scores[chunk_string] + math_score
            rank = rank + 1

    def grab_the_score(dictionary_item):
        return dictionary_item[1] 
    sorted_scores = sorted(rrf_scores.items(), key=grab_the_score, reverse=True)
    final_fused_chunks = []
    for item in sorted_scores:
        final_fused_chunks.append(item[0])
    return final_fused_chunks
result__final=reciprocal_rank_fusion(result_final)
result2="\n\n".join(result__final)
SYSTEM_PROMPT2=f""
"YOU ARE A AI AGENT WHICH HANDLES USER QUERY WITH THE HELP OF THE GIVEN DATA BELOW"
{result2}
"- RULE FOR FINAL OUTPUT: the Content MUST be formatted as clean, readable Markdown text (using bullet points). DO NOT output nested JSON, dictionaries, or raw arrays in the final content."
""
client2 = genai.Client(api_key="")
content=client2.models.generate_content(
     model="gemini-2.5-flash", # Using flash for better reasoning capability
     contents="EXPLAIN THE INSTRUCTION CYCLE?",
    config=types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT2,
        temperature=0.2,
    ),
)
with open(f"files_desc.md", "w") as file:
            file.write(f"{content.text}")
