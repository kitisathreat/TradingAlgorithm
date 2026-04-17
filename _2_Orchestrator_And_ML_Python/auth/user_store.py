"""User credential storage backed by a YAML file on disk.

Passwords are bcrypt-hashed. The YAML format is compatible with
`streamlit-authenticator` so the same file can be fed directly into
`stauth.Authenticate(...)`.
"""

from __future__ import annotations

import logging
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import bcrypt
import yaml

logger = logging.getLogger(__name__)

DEFAULT_STORE = Path("data") / "users.yaml"
DEFAULT_USER_ROOT = Path("data") / "users"


@dataclass
class UserRecord:
    username: str
    name: str
    email: str
    password_hash: str


class UserStore:
    """YAML-backed user credential store."""

    def __init__(
        self,
        path: Path = DEFAULT_STORE,
        user_root: Path = DEFAULT_USER_ROOT,
    ) -> None:
        self.path = Path(path)
        self.user_root = Path(user_root)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.user_root.mkdir(parents=True, exist_ok=True)

    def load(self) -> Dict:
        if not self.path.exists():
            return self._empty_config()
        with self.path.open("r") as f:
            data = yaml.safe_load(f) or {}
        if "cookie" not in data:
            data["cookie"] = self._default_cookie()
        if "credentials" not in data:
            data["credentials"] = {"usernames": {}}
        return data

    def save(self, config: Dict) -> None:
        with self.path.open("w") as f:
            yaml.safe_dump(config, f)

    def add_user(self, username: str, name: str, password: str, email: str = "") -> bool:
        username = username.strip().lower()
        if not username or not password:
            return False
        config = self.load()
        users = config["credentials"]["usernames"]
        if username in users:
            return False
        users[username] = {
            "email": email or f"{username}@example.com",
            "name": name or username,
            "password": self.hash_password(password),
        }
        self.save(config)
        self.user_dir(username)
        logger.info("Created user '%s'", username)
        return True

    def verify_password(self, username: str, password: str) -> bool:
        username = username.strip().lower()
        config = self.load()
        user = config["credentials"]["usernames"].get(username)
        if not user:
            return False
        try:
            return bcrypt.checkpw(password.encode("utf-8"), user["password"].encode("utf-8"))
        except ValueError:
            return False

    def change_password(self, username: str, new_password: str) -> bool:
        username = username.strip().lower()
        config = self.load()
        user = config["credentials"]["usernames"].get(username)
        if not user:
            return False
        user["password"] = self.hash_password(new_password)
        self.save(config)
        return True

    def user_exists(self, username: str) -> bool:
        config = self.load()
        return username.strip().lower() in config["credentials"]["usernames"]

    def list_users(self) -> list[str]:
        config = self.load()
        return sorted(config["credentials"]["usernames"].keys())

    def user_dir(self, username: str) -> Path:
        username = username.strip().lower()
        base = self.user_root / username
        for sub in ("model", "trades", "training", "fundamentals_cache", "debug"):
            (base / sub).mkdir(parents=True, exist_ok=True)
        return base

    @staticmethod
    def hash_password(password: str) -> str:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    def _empty_config(self) -> Dict:
        return {"credentials": {"usernames": {}}, "cookie": self._default_cookie()}

    @staticmethod
    def _default_cookie() -> Dict:
        key = os.environ.get("TRADING_COOKIE_KEY") or secrets.token_hex(16)
        return {"name": "trading_auth_cookie", "key": key, "expiry_days": 7}
