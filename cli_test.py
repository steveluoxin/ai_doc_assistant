from llm import call_llm
from rag import build_trunks_and_embeddings, analyze_document_structure, retrieve_for_summary

def load_document(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
    
document = load_document("prd.txt")

structure = analyze_document_structure(document)
DOCUMENT_STRUCTURE = structure
print("结构分析结果：")
print(DOCUMENT_STRUCTURE)

chunks, embeddings = build_trunks_and_embeddings(document)

# for i, chunk in enumerate(chunks):
#     print(f"---- chunk {i} ----")
#     print(chunk)


from rag import retrieve_top_k_chunks_with_score, embed_query, answer_with_rag, rewrite_query

query = "在这个产品文档里，考试系统的核心功能有哪些？"
task = "summary"

rewritten_query = rewrite_query(query, structure=DOCUMENT_STRUCTURE)
print("改写后的检索问题: ", rewritten_query)
if task == "summary":
    results = retrieve_for_summary(
        question=query,
        structure=DOCUMENT_STRUCTURE,
        chunks=chunks,
        embeddings=embeddings,
        per_module_k=3
    )
else:
    query_emb = embed_query(rewritten_query)
    results = retrieve_top_k_chunks_with_score(...)
# for chunk, score in results:
#     print("Score: ", round(score, 3))
#     print(chunk)
#     print("-----")

result = answer_with_rag(
    question=query, retrieved_chunks=results, task=task
)
print(result)



