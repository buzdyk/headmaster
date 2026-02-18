import numpy as np
import pytest
import torch

from headmaster import db
from headmaster.embed import serialize_vector, deserialize_vector


class TestModelRegistry:
    def test_add_and_list(self, workspace):
        db.model_add(workspace, "clip", "openai/clip", 768)
        models = db.model_list(workspace)
        assert len(models) == 1
        assert models[0]["name"] == "clip"
        assert models[0]["path"] == "openai/clip"
        assert models[0]["embed_dim"] == 768
        assert models[0]["active"] == 0

    def test_add_duplicate_exits(self, workspace):
        db.model_add(workspace, "clip", "openai/clip", 768)
        with pytest.raises(SystemExit, match="already exists"):
            db.model_add(workspace, "clip", "openai/clip", 768)

    def test_activate(self, workspace):
        db.model_add(workspace, "clip", "openai/clip", 768)
        db.model_add(workspace, "dino", "fb/dino", 1024)
        db.model_activate(workspace, "clip")

        models = db.model_list(workspace)
        active = [m for m in models if m["active"]]
        assert len(active) == 1
        assert active[0]["name"] == "clip"

    def test_activate_switches(self, workspace):
        db.model_add(workspace, "clip", "openai/clip", 768)
        db.model_add(workspace, "dino", "fb/dino", 1024)
        db.model_activate(workspace, "clip")
        db.model_activate(workspace, "dino")

        active = db.get_active_model(workspace)
        assert active["name"] == "dino"

    def test_activate_nonexistent_exits(self, workspace):
        with pytest.raises(SystemExit, match="not found"):
            db.model_activate(workspace, "nope")

    def test_remove(self, workspace):
        db.model_add(workspace, "clip", "openai/clip", 768)
        db.model_remove(workspace, "clip")
        assert db.model_list(workspace) == []

    def test_remove_nonexistent_exits(self, workspace):
        with pytest.raises(SystemExit, match="not found"):
            db.model_remove(workspace, "nope")

    def test_remove_deletes_embeddings(self, workspace):
        db.model_add(workspace, "clip", "openai/clip", 768)
        model = db.get_active_model(workspace) or {"id": 1}
        # We need to get the model id
        models = db.model_list(workspace)
        # Get id via a separate query since model_list doesn't return id
        db.model_activate(workspace, "clip")
        model = db.get_active_model(workspace)

        vec = serialize_vector(torch.randn(768))
        db.put_embedding(workspace, "abc123", model["id"], vec)
        assert db.get_cached_hashes(workspace, model["id"]) == {"abc123"}

        db.model_remove(workspace, "clip")
        assert db.get_cached_hashes(workspace, model["id"]) == set()

    def test_get_active_model_none(self, workspace):
        assert db.get_active_model(workspace) is None

    def test_get_active_model(self, workspace):
        db.model_add(workspace, "clip", "openai/clip", 768)
        db.model_activate(workspace, "clip")
        m = db.get_active_model(workspace)
        assert m["name"] == "clip"
        assert m["embed_dim"] == 768

    def test_list_ordering(self, workspace):
        db.model_add(workspace, "b-model", "path/b", 512)
        db.model_add(workspace, "a-model", "path/a", 256)
        models = db.model_list(workspace)
        assert models[0]["name"] == "b-model"
        assert models[1]["name"] == "a-model"


class TestEmbeddingCache:
    def test_put_and_get(self, workspace_with_model):
        ws = workspace_with_model
        model = db.get_active_model(ws)
        vec = torch.randn(64)
        db.put_embedding(ws, "hash1", model["id"], serialize_vector(vec))

        data = db.get_embedding(ws, "hash1", model["id"])
        assert data is not None
        recovered = deserialize_vector(data, 64)
        assert torch.allclose(vec, recovered)

    def test_get_missing(self, workspace_with_model):
        ws = workspace_with_model
        model = db.get_active_model(ws)
        assert db.get_embedding(ws, "nonexistent", model["id"]) is None

    def test_get_cached_hashes(self, workspace_with_model):
        ws = workspace_with_model
        model = db.get_active_model(ws)
        for h in ["aaa", "bbb", "ccc"]:
            db.put_embedding(ws, h, model["id"], serialize_vector(torch.randn(64)))

        cached = db.get_cached_hashes(ws, model["id"])
        assert cached == {"aaa", "bbb", "ccc"}

    def test_put_replaces_existing(self, workspace_with_model):
        ws = workspace_with_model
        model = db.get_active_model(ws)
        vec1 = torch.ones(64)
        vec2 = torch.zeros(64)
        db.put_embedding(ws, "hash1", model["id"], serialize_vector(vec1))
        db.put_embedding(ws, "hash1", model["id"], serialize_vector(vec2))

        data = db.get_embedding(ws, "hash1", model["id"])
        recovered = deserialize_vector(data, 64)
        assert torch.allclose(recovered, vec2)

    def test_embeddings_scoped_to_model(self, workspace):
        db.model_add(workspace, "m1", "path1", 64)
        db.model_add(workspace, "m2", "path2", 64)
        db.model_activate(workspace, "m1")
        m1 = db.get_active_model(workspace)
        db.model_activate(workspace, "m2")
        m2 = db.get_active_model(workspace)

        db.put_embedding(workspace, "hash1", m1["id"], serialize_vector(torch.randn(64)))
        assert db.get_cached_hashes(workspace, m1["id"]) == {"hash1"}
        assert db.get_cached_hashes(workspace, m2["id"]) == set()
