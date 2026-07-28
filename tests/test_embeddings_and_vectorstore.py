from tests import _env  # noqa: F401  must be first

import unittest

import numpy as np

from app.embeddings import get_embedder
from app.vectorstore import LocalVectorStore


class TestEmbeddings(unittest.TestCase):
    def test_embeddings_are_l2_normalized(self):
        emb = get_embedder()
        vecs = emb.embed_texts(["hello world", "another sentence here"])
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-5)

    def test_related_texts_more_similar_than_unrelated(self):
        emb = get_embedder()
        a, b, c = emb.embed_texts(
            [
                "Sparse beamforming reduces complexity in massive MIMO systems.",
                "Beamforming algorithms for large antenna arrays.",
                "Photosynthesis converts sunlight into chemical energy.",
            ]
        )
        self.assertGreater(a @ b, a @ c)

    def test_empty_string_does_not_crash(self):
        emb = get_embedder()
        vecs = emb.embed_texts([""])
        self.assertEqual(vecs.shape[0], 1)


class TestVectorStore(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path

        self.store = LocalVectorStore(persist_dir=Path(tempfile.mkdtemp()))
        emb = get_embedder()
        docs = [
            ("c1", "p1", 1, "Sparse beamforming reduces complexity."),
            ("c2", "p1", 2, "Greedy pursuit suffers from local optima."),
            ("c3", "p2", 1, "Photosynthesis happens in chloroplasts."),
        ]
        vecs = emb.embed_texts([d[3] for d in docs])
        items = [{"chunk_id": d[0], "paper_id": d[1], "page_number": d[2], "text": d[3]} for d in docs]
        self.store.add_batch(items, vecs)
        self.emb = emb

    def test_count(self):
        self.assertEqual(self.store.count(), 3)

    def test_search_ranks_relevant_result_first(self):
        q = self.emb.embed_texts(["What reduces algorithm complexity?"])[0]
        results = self.store.search(q, top_k=3)
        self.assertEqual(results[0].chunk_id, "c1")

    def test_paper_id_filter_restricts_results(self):
        q = self.emb.embed_texts(["anything"])[0]
        results = self.store.search(q, top_k=5, paper_ids=["p2"])
        self.assertTrue(all(r.paper_id == "p2" for r in results))
        self.assertEqual(len(results), 1)

    def test_delete_paper_removes_only_its_chunks(self):
        removed = self.store.delete_paper("p1")
        self.assertEqual(removed, 2)
        self.assertEqual(self.store.count(), 1)

    def test_persistence_round_trip(self):
        persist_dir = self.store._dir
        reloaded = LocalVectorStore(persist_dir=persist_dir)
        self.assertEqual(reloaded.count(), 3)


if __name__ == "__main__":
    unittest.main()
