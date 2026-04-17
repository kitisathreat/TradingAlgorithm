"""Tests for auth.user_store."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "_2_Orchestrator_And_ML_Python"))
import pytest

def test_add_user_creates_entry(user_store_tmp):
    ok = user_store_tmp.add_user("alice", "Alice", "secret123")
    assert ok
    cfg = user_store_tmp.load()
    assert "alice" in cfg["credentials"]["usernames"]

def test_password_is_hashed(user_store_tmp):
    import bcrypt
    user_store_tmp.add_user("bob", "Bob", "mypassword")
    cfg = user_store_tmp.load()
    stored = cfg["credentials"]["usernames"]["bob"]["password"].encode()
    assert bcrypt.checkpw(b"mypassword", stored)

def test_duplicate_user_rejected(user_store_tmp):
    user_store_tmp.add_user("carol", "Carol", "pw1")
    ok = user_store_tmp.add_user("carol", "Carol", "pw2")
    assert not ok

def test_user_dir_created(user_store_tmp, tmp_path):
    user_store_tmp.add_user("dave", "Dave", "pw")
    d = user_store_tmp.user_dir("dave")
    assert (d / "model").exists()
    assert (d / "trades").exists()
    assert (d / "training").exists()

def test_load_returns_default_on_missing(tmp_path):
    from auth.user_store import UserStore
    store = UserStore(path=tmp_path / "nonexistent.yaml")
    cfg = store.load()
    assert "credentials" in cfg
    assert "usernames" in cfg["credentials"]
