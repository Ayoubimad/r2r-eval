from chonkie.chunker import SemanticChunker, SDPMChunker
from generic_embeddings import GenericEmbeddings
from chonkie.embeddings import BaseEmbeddings


embedding_model = GenericEmbeddings(
    model="BAAI/bge-m3",
    api_key="sk-proj-1234567890",
    base_url="http://172.18.21.126:8000/v1",
    embedding_dimension=1024,
    timeout=60000,
)

semantic_chunker = SemanticChunker(
    embedding_model=embedding_model,
    chunk_size=256,
)

sdp_chunker = SDPMChunker(
    embedding_model=embedding_model,
    chunk_size=256,
    min_chunk_size=8,
)


chunks = semantic_chunker.chunk("Hello, world!")

print(chunks)

chunks = sdp_chunker.chunk("Hello, world!")

print(chunks)
