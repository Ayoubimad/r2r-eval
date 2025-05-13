from config import IngestionConfig, ChunkEnrichmentSettings, RAGGenerationConfig
from r2r import R2RClient
from chunk_enrichment import ChunkEnrichmentStrategy

client = R2RClient("http://localhost:7272")


documents = client.documents.list(limit=1000)
for doc in documents.results:
    try:
        client.documents.delete(str(doc.id))
    except Exception as e:
        print(f"Error deleting document {doc.id}")


generation_config = RAGGenerationConfig(
    model="deepseek/ISTA-DASLab/gemma-3-4b-it-GPTQ-4b-128g",
    api_base="http://172.18.21.136:8000/v1/",
    temperature=0.8,
    max_tokens=50000,
)

chunks = [
    "The Transformer architecture revolutionized natural language processing by introducing self-attention mechanisms that capture long-range dependencies in text. Self-attention allows each position in a sequence to attend to all positions, enabling the model to weigh the importance of different words regardless of their distance. Unlike recurrent neural networks, Transformers process entire sequences in parallel, making them more efficient for training on large datasets. Modern large language models build upon the Transformer architecture, scaling to billions of parameters to improve their ability to understand and generate human language.",
    "The key innovation of Transformers is their ability to process input sequences in parallel rather than sequentially like RNNs. This parallelization is achieved through the self-attention mechanism, which creates a weighted sum of all input elements for each position. The weights are computed using queries, keys and values derived from the input embeddings. This allows the model to capture both local and global dependencies without regard to sequential distance. The multi-head attention mechanism further improves this by allowing the model to attend to different representation subspaces simultaneously.",
    "Transformers have enabled breakthrough advances in natural language processing by addressing fundamental limitations of previous architectures. Their parallel processing capability allows efficient training on massive datasets. The self-attention mechanism captures intricate relationships between words regardless of their distance in the sequence. The architecture scales effectively to very large model sizes, leading to increasingly sophisticated language understanding capabilities. These innovations have enabled the development of powerful language models that can perform a wide range of linguistic tasks with unprecedented effectiveness.",
]

chunk_enrichment = ChunkEnrichmentStrategy(
    n_chunks=1,
    model="ISTA-DASLab/gemma-3-4b-it-GPTQ-4b-128g",
    api_base="http://172.18.21.136:8000/v1/",
    api_key="random-key",
    max_tokens=50000,
    num_workers=32,
)

print("\n📚 Enriching small batch of chunks...")
enriched_chunks = chunk_enrichment.enrich_chunks(chunks)

print("\n📚 Original vs Enriched Chunks:")
for i, (original, enriched) in enumerate(zip(chunks, enriched_chunks)):
    print(f"\n✨ Chunk {i} ✨")
    print(f"\nOriginal:\n{original}\n")
    print(f"Enriched:\n{enriched}\n")
    print(f"Equal: {original == enriched}")
    print(f"Original length: {len(original)} | Enriched length: {len(enriched)}")


with open("/home/e4user/r2r-eval/data/output/diskann_neurips19.md", "r") as f:
    doc = f.read()

from chunking import LangChainChunking

chunker = LangChainChunking(
    splitter_type="character", chunk_size=1024, chunk_overlap=250
)
doc = chunker.clean_text(doc)
from document import Document

chunks = chunker.chunk(Document(content=doc))

chunks = [c.content for c in chunks]

print("Enriching larger batch of chunks...")
enriched_chunks = chunk_enrichment.enrich_chunks(chunks)

print("\n📚 Original vs Enriched Chunks:")
for i, (original, enriched) in enumerate(zip(chunks, enriched_chunks)):
    print(f"\n✨ Chunk {i} ✨")
    print(f"\nOriginal:\n{original}\n")
    print(f"Enriched:\n{enriched}\n")
    print(f"Equal: {original == enriched}")
    print(f"Original length: {len(original)} | Enriched length: {len(enriched)}")

exit()
document = client.documents.create(raw_text=doc)
# document = client.documents.create(chunks=chunks)

# exit()

query = "How do transformer models process text differently from RNNs?"
rag_response = client.retrieval.rag(
    query=query,
    search_mode="basic",
    search_settings={"limit": 5, "graph_settings": {"enabled": False}},
)

print(f"Query: {query}")
print(f"Generated answer: {rag_response.results.generated_answer}")

print("\nRetrieved chunks:")
retrieved_chunks = []
for i, result in enumerate(rag_response.results.search_results.chunk_search_results, 1):
    print(f"{i}. Score: {result.score:.3f} | Text: {result.text}")
    retrieved_chunks.append(result.text)
