from abc import ABC, abstractmethod
from pathlib import Path


class BaseTokenizer(ABC):
    name: str

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        pass

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        pass

    @abstractmethod
    def save(self, path: str | Path) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path):
        pass