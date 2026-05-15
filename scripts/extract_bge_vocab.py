#!/usr/bin/env python3
"""Extract BGE tokenizer vocab and config for Swift WordPiece tokenization.

Outputs:
    GuidelineAssistant/GuidelineAssistant/Resources/bge_vocab.txt
    GuidelineAssistant/GuidelineAssistant/Resources/bge_tokenizer_config.json
"""

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer


MODEL_NAME = "BAAI/bge-small-en-v1.5"
RESOURCES_DIR = Path("GuidelineAssistant/GuidelineAssistant/Resources")
VOCAB_PATH = RESOURCES_DIR / "bge_vocab.txt"
CONFIG_PATH = RESOURCES_DIR / "bge_tokenizer_config.json"


def _resolve_max_length(tokenizer: AutoTokenizer) -> int:
    value = int(getattr(tokenizer, "model_max_length", 512))
    # Hugging Face often uses very large sentinel values for "unbounded".
    if value <= 0 or value > 100000:
        return 512
    return value


def main() -> None:
    RESOURCES_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    vocab = tokenizer.get_vocab()

    sorted_tokens = sorted(vocab.items(), key=lambda item: item[1])
    with VOCAB_PATH.open("w", encoding="utf-8") as handle:
        for token, _ in sorted_tokens:
            handle.write(f"{token}\n")

    config = {
        "cls_token_id": int(tokenizer.cls_token_id),
        "sep_token_id": int(tokenizer.sep_token_id),
        "pad_token_id": int(tokenizer.pad_token_id),
        "unk_token_id": int(tokenizer.unk_token_id),
        "max_length": _resolve_max_length(tokenizer),
        "do_lower_case": bool(tokenizer.init_kwargs.get("do_lower_case", False)),
    }

    with CONFIG_PATH.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    line_count = len(sorted_tokens)
    first_token = sorted_tokens[0][0] if sorted_tokens else ""
    print("Tokenizer extraction complete")
    print(f"  Vocab path: {VOCAB_PATH}")
    print(f"  Config path: {CONFIG_PATH}")
    print(f"  Vocabulary entries: {line_count}")
    print(f"  Token id 0: {first_token}")


if __name__ == "__main__":
    main()
