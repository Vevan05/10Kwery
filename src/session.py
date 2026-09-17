from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConversationSession:


    turns: list[dict] = field(default_factory=list)
    last_tickers: list[str] = field(default_factory=list)
    last_chunks: list[dict] = field(default_factory=list)

    def add_user(self, content: str, tickers: list[str] | None = None) -> None:
        self.turns.append(
            {"role": "user", "content": content, "tickers": tickers or []}
        )
        if tickers:
            self.last_tickers = list(tickers)

    def add_assistant(self, content: str) -> None:
        self.turns.append(
            {"role": "assistant", "content": content, "tickers": list(self.last_tickers)}
        )

    def history_for_retrieval(self, n: int = 4) -> list[dict]:
        return self.turns[-n:]

    def prompt_context(self, n: int = 2) -> str:

        if not self.turns:
            return ""
        last = self.turns[-2 * n :]
        lines = [f"{t['role']}: {t['content'][:400]}" for t in last]
        return "Conversation so far:\n" + "\n".join(lines) + "\n\n"
