import argparse
from rag_engine import rebuild_index
from config import CHUNK_SIZE, CHUNK_OVERLAP

parser = argparse.ArgumentParser(description="Build/rebuild the local vector index from knowledge_base documents.")
parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
args = parser.parse_args()

count = rebuild_index(args.chunk_size, args.overlap)
print(f"Index rebuilt successfully: {count} chunks stored.")