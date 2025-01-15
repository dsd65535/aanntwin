"""This module containts utility functions"""
from hashlib import blake2b


def hash_str(input_string: str, num_bytes: int = 4) -> str:
    """A repeatable, insecure hash for string compression"""

    hasher = blake2b(digest_size=num_bytes, usedforsecurity=False)
    hasher.update(input_string.encode("utf-8"))

    return hasher.hexdigest()
