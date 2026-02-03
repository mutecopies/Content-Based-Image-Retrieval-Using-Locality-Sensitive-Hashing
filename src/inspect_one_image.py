import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from core.vector_db import VectorDatabase

db = VectorDatabase(
    dim=512,
    persist_path="data/embeddings/caltech101_db.pkl",
    use_lsh=True,
    lsh_params={'num_tables': 8, 'hash_size': 10, 'seed': 42}
)

meta = db.get_metadata("img_01784")
print(meta)
