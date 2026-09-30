from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import torch


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    normalized = str(text).strip().lower()
    if not normalized:
        return []
    return TOKEN_PATTERN.findall(normalized)


class WordTokenizer:
    def __init__(self) -> None:
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        self.id_to_token: list[str] = [
            self.pad_token,
            self.bos_token,
            self.eos_token,
            self.unk_token,
        ]
        self.token_to_id: dict[str, int] = {token: index for index, token in enumerate(self.id_to_token)}

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.eos_token]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]

    def fit(self, texts: list[str]) -> None:
        for text in texts:
            for token in tokenize(text):
                if token not in self.token_to_id:
                    self.token_to_id[token] = len(self.id_to_token)
                    self.id_to_token.append(token)

    def encode(
        self,
        text: str,
        *,
        max_length: int,
        add_bos: bool = False,
        add_eos: bool = True,
    ) -> list[int]:
        token_ids: list[int] = []
        if add_bos:
            token_ids.append(self.bos_id)
        token_ids.extend(self.token_to_id.get(token, self.unk_id) for token in tokenize(text))
        if add_eos:
            token_ids.append(self.eos_id)
        token_ids = token_ids[:max_length]
        if len(token_ids) < max_length:
            token_ids.extend([self.pad_id] * (max_length - len(token_ids)))
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        tokens: list[str] = []
        for token_id in token_ids:
            if token_id == self.pad_id:
                continue
            if token_id == self.eos_id:
                break
            if token_id == self.bos_id:
                continue
            tokens.append(self.id_to_token[token_id])
        return " ".join(tokens).strip()


@dataclass
class VocabularySet:
    text_tokenizer: WordTokenizer
    commit_to_id: dict[str, int]
    id_to_commit: list[str]
    subtask_to_id: dict[str, int]
    id_to_subtask: list[str]
    subtask_texts: list[str]
    memory_texts: list[str]
    max_task_tokens: int
    max_memory_tokens: int
    max_subtask_tokens: int

    @property
    def no_update_id(self) -> int:
        return self.commit_to_id["no_update"]

    def encode_task(self, text: str) -> list[int]:
        return self.text_tokenizer.encode(text, max_length=self.max_task_tokens, add_bos=False, add_eos=True)

    def encode_memory(self, text: str) -> list[int]:
        return self.text_tokenizer.encode(text, max_length=self.max_memory_tokens, add_bos=False, add_eos=True)

    def encode_subtask(self, text: str) -> list[int]:
        return self.text_tokenizer.encode(text, max_length=self.max_subtask_tokens, add_bos=False, add_eos=True)

    def memory_bank_token_tensor(self) -> torch.Tensor:
        bank = [self.encode_memory(text) for text in self.memory_texts]
        return torch.tensor(bank, dtype=torch.long)

    def subtask_bank_token_tensor(self) -> torch.Tensor:
        bank = [self.encode_subtask(text) for text in self.subtask_texts]
        return torch.tensor(bank, dtype=torch.long)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokenizer_tokens": self.text_tokenizer.id_to_token,
            "commit_to_id": self.commit_to_id,
            "id_to_commit": self.id_to_commit,
            "subtask_to_id": self.subtask_to_id,
            "id_to_subtask": self.id_to_subtask,
            "subtask_texts": self.subtask_texts,
            "memory_texts": self.memory_texts,
            "max_task_tokens": self.max_task_tokens,
            "max_memory_tokens": self.max_memory_tokens,
            "max_subtask_tokens": self.max_subtask_tokens,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VocabularySet":
        tokenizer = WordTokenizer()
        tokenizer.id_to_token = list(payload["tokenizer_tokens"])
        tokenizer.token_to_id = {token: index for index, token in enumerate(tokenizer.id_to_token)}
        return cls(
            text_tokenizer=tokenizer,
            commit_to_id={str(k): int(v) for k, v in payload["commit_to_id"].items()},
            id_to_commit=[str(item) for item in payload["id_to_commit"]],
            subtask_to_id={str(k): int(v) for k, v in payload["subtask_to_id"].items()},
            id_to_subtask=[str(item) for item in payload["id_to_subtask"]],
            subtask_texts=[str(item) for item in payload["subtask_texts"]],
            memory_texts=[str(item) for item in payload["memory_texts"]],
            max_task_tokens=int(payload["max_task_tokens"]),
            max_memory_tokens=int(payload["max_memory_tokens"]),
            max_subtask_tokens=int(payload["max_subtask_tokens"]),
        )


def build_vocabulary(rows: list[dict[str, Any]], config: dict[str, Any]) -> VocabularySet:
    tokenizer = WordTokenizer()
    all_texts: list[str] = []
    commit_labels = {"no_update"}
    subtask_texts: set[str] = set()
    memory_texts: set[str] = set()

    for row in rows:
        task_text = str(row["task"])
        prev_memory = str(row["previous_memory_text"])
        memory_text = str(row["memory_text"])
        subtask_text = str(row["subtask"])
        all_texts.extend([task_text, prev_memory, memory_text, subtask_text])
        memory_texts.add(memory_text)
        memory_texts.add(prev_memory)
        subtask_texts.add(subtask_text)
        if row.get("update_required", True):
            commit_labels.add(str(row.get("update_trigger", "state_change")))
        else:
            commit_labels.add("no_update")

    tokenizer.fit(all_texts)
    sorted_commits = ["no_update"] + sorted(label for label in commit_labels if label != "no_update")
    sorted_subtasks = sorted(subtask_texts)
    sorted_memories = sorted(memory_texts)

    return VocabularySet(
        text_tokenizer=tokenizer,
        commit_to_id={label: index for index, label in enumerate(sorted_commits)},
        id_to_commit=sorted_commits,
        subtask_to_id={label: index for index, label in enumerate(sorted_subtasks)},
        id_to_subtask=sorted_subtasks,
        subtask_texts=sorted_subtasks,
        memory_texts=sorted_memories,
        max_task_tokens=int(config["max_task_tokens"]),
        max_memory_tokens=int(config["max_memory_tokens"]),
        max_subtask_tokens=int(config["max_subtask_tokens"]),
    )

