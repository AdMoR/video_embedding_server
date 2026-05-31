from abc import ABC, abstractmethod


class VideoCaptioner(ABC):
    @abstractmethod
    def load_model(self) -> None: ...

    @abstractmethod
    def caption(self, path: str) -> dict: ...

    @abstractmethod
    def caption_with_prompt(self, path: str, prompt: str, max_new_tokens: int = 512) -> str: ...

    @abstractmethod
    def find(self, path: str, event: str) -> dict: ...
