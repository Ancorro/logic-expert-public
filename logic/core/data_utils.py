"""Dataset loaders with deterministic splits for logic-stream benchmarks."""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def _validate_max_samples(max_samples: Optional[int], dataset_len: int) -> None:
    """Validate optional sample cap against dataset size and positivity."""
    if max_samples is None:
        return
    if max_samples <= 0:
        raise ValueError(f"max_samples must be positive when set, got {max_samples}")
    if max_samples > dataset_len:
        raise ValueError(
            f"max_samples ({max_samples}) exceeds dataset size ({dataset_len}); "
            "set null to use the full split"
        )


def _require_dict(config: dict[str, Any], key: str) -> dict[str, Any]:
    """Fetch a required config sub-block and assert it is a dict."""
    value = config.get(key)
    if not isinstance(value, dict):
        raise KeyError(f"Missing required config block '{key}'")
    return value


def _require_keys(block: dict[str, Any], block_name: str, keys: list[str]) -> None:
    """Assert that a config block contains all required keys."""
    missing = [k for k in keys if k not in block]
    if missing:
        raise KeyError(f"Missing required keys in {block_name}: {missing}")


def get_tokenizer(model_name: str):
    """Load tokenizer and guarantee a usable pad token for batched padding."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError(
                f"Tokenizer for '{model_name}' has no pad_token or eos_token; "
                "configure a pad token before using padding=True."
            )
    return tokenizer


def load_glue_cola(
    model_name: str = "meta-llama/Meta-Llama-3.1-8B",
    max_length: int = 128,
    split: str = "train",
    max_samples: Optional[int] = None,
    seed: int = 42,
):
    """Load GLUE CoLA split with deterministic optional subsampling.

    Returns dataset object and collate function that tokenizes sentence text and
    emits ``input_ids``, ``attention_mask``, and ``labels`` tensors.
    """
    from datasets import load_dataset

    ds = load_dataset("glue", "cola", split=split)
    _validate_max_samples(max_samples, len(ds))
    if max_samples:
        import random
        rng = random.Random(seed)
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        ds = ds.select(indices[:max_samples])

    tokenizer = get_tokenizer(model_name)

    def collate(examples):
        batch = tokenizer(
            [ex["sentence"] for ex in examples],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor([ex["label"] for ex in examples])
        return dict(batch)

    return ds, collate


def load_proofwriter(
    model_name: str = "meta-llama/Meta-Llama-3.1-8B",
    max_length: int = 256,
    depth: str = "depth-0",
    split: str = "train",
    config_name: str | None = "default",
    max_samples: Optional[int] = None,
    seed: int = 42,
):
    """Load ProofWriter split with deterministic optional subsampling.

    Handles common schema variants by detecting text/label columns and normalizing
    labels to integer class values.
    """
    from datasets import load_dataset

    # Prefer tasksource/proofwriter (logic_hf-style), with longface fallback.
    full = None
    load_errors = []
    candidates = [
        ("tasksource/proofwriter", config_name),
        ("longface/ProofWriter", None),
    ]
    loaded_dataset_name = None

    for dataset_name, dataset_cfg in candidates:
        try:
            full = load_dataset(dataset_name, dataset_cfg) if dataset_cfg else load_dataset(dataset_name)
            loaded_dataset_name = dataset_name
            break
        except Exception as e:
            load_errors.append(f"{dataset_name}({dataset_cfg}): {type(e).__name__}: {e}")

    if full is None:
        raise RuntimeError(
            "Failed to load ProofWriter dataset from known sources. Errors: "
            + " | ".join(load_errors)
        )

    if split not in full.keys():
        available = sorted(list(full.keys()))
        raise ValueError(
            f"Requested split '{split}' is unavailable for {loaded_dataset_name}. "
            f"Available splits: {available}"
        )
    ds = full[split]

    # Keep all depths by default; optionally filter if an explicit depth is requested.
    depth_normalized = str(depth).strip().lower()
    if depth_normalized not in ("all", "*"):
        depth_digits = "".join(ch for ch in depth_normalized if ch.isdigit())
        if not depth_digits:
            raise ValueError(f"Invalid depth value '{depth}'. Use 'all' or forms like 'depth-2'.")
        target_depth = int(depth_digits)

        def _extract_depth(ex):
            for key in ("depth", "proof_depth", "metadata"):
                if key in ex and ex[key] is not None:
                    value = ex[key]
                    if isinstance(value, int):
                        return value
                    text = str(value)
                    digits = "".join(ch for ch in text if ch.isdigit())
                    if digits:
                        return int(digits)
            return None

        ds = ds.filter(lambda ex: _extract_depth(ex) == target_depth)

    _validate_max_samples(max_samples, len(ds))

    if max_samples and len(ds) > max_samples:
        import random
        rng = random.Random(seed)
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        ds = ds.select(indices[:max_samples])

    # ProofWriter schema varies across sources. Prefer a direct text column,
    # otherwise synthesize one from context/facts/rules + question fields.
    if "input" in ds.column_names:
        def to_text(ex):
            return str(ex["input"])
    elif "context" in ds.column_names:
        def to_text(ex):
            context = str(ex["context"]).strip()
            question = str(ex["question"]).strip() if "question" in ex else ""
            if context and question:
                return f"{context}\nQuestion: {question}"
            return context if context else str(ex)
    elif any(c in ds.column_names for c in ("facts", "rules", "question")):
        def to_text(ex):
            facts = str(ex["facts"]).strip() if "facts" in ex else ""
            rules = str(ex["rules"]).strip() if "rules" in ex else ""
            question = str(ex["question"]).strip() if "question" in ex else ""

            parts = []
            if facts:
                parts.append(f"Facts: {facts}")
            if rules:
                parts.append(f"Rules: {rules}")
            if question:
                parts.append(f"Question: {question}")
            return "\n".join(parts) if parts else str(ex)
    else:
        raise KeyError(
            "ProofWriter text content not found. Expected one of "
            "['input', 'context', 'facts/rules/question']; dataset columns: "
            f"{ds.column_names}"
        )
    if "answer" in ds.column_names:
        def _normalize_proofwriter_answer(answer: Any) -> int:
            """Normalize ProofWriter answers into 3-class ids: 0=false, 1=true, 2=unknown."""
            if isinstance(answer, bool):
                return 1 if answer else 0

            if isinstance(answer, (int, float)) and not isinstance(answer, bool):
                val = int(answer)
                if val in (0, 1, 2):
                    return val
                if val in (1, 2, 3):
                    return val - 1

            text = str(answer).strip().lower()
            if text in ("false", "no", "0", "contradiction", "refuted", "refutes"):
                return 0
            if text in ("true", "yes", "1", "entailment", "entailed", "supports", "provable"):
                return 1
            if text in ("unknown", "uncertain", "both", "indeterminate", "neither", "inconclusive"):
                return 2

            raise ValueError(f"Unrecognized ProofWriter answer label: {answer!r}")

        def to_label(ex):
            return _normalize_proofwriter_answer(ex["answer"])
    else:
        if "label" not in ds.column_names:
            raise KeyError(
                f"ProofWriter dataset missing both 'answer' and 'label' columns: {ds.column_names}"
            )

        def to_label(ex):
            return int(ex["label"])

    tokenizer = get_tokenizer(model_name)

    def collate(examples):
        texts = [to_text(ex) for ex in examples]
        labels = torch.tensor([to_label(ex) for ex in examples])
        batch = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch["labels"] = labels
        return dict(batch)

    return ds, collate


def build_dataloader(
    config: dict[str, Any],
    split: str = "train",
) -> tuple[DataLoader, Callable]:
    """Build a configured ``DataLoader`` and matching collate function.

    Supports:
    - ``proofwriter`` via dedicated loader
    - generic HuggingFace dataset/config pairs using ``data.text_col`` and
        ``data.label_col``

    Returns:
        ``(loader, collate_fn)`` where ``loader`` is configured for the requested
        split and ``collate_fn`` can be reused by evaluation code.
    """
    data_cfg = _require_dict(config, "data")
    model_cfg = _require_dict(config, "model")
    train_cfg = _require_dict(config, "train")

    _require_keys(data_cfg, "data", ["dataset", "config", "max_length"])
    _require_keys(model_cfg, "model", ["model_name"])
    _require_keys(train_cfg, "train", ["seed", "batch_size"])

    dataset_name = data_cfg["dataset"]
    dataset_config = data_cfg["config"]
    max_length = data_cfg["max_length"]
    max_samples = data_cfg.get("max_samples")
    seed = train_cfg["seed"]
    model_name = model_cfg["model_name"]

    if dataset_name.lower() == "proofwriter":
        if "depth" not in data_cfg:
            raise KeyError("Missing required key data.depth for proofwriter dataset")
        ds, collate = load_proofwriter(
            model_name=model_name,
            max_length=max_length,
            depth=data_cfg["depth"],
            split=split,
            config_name=dataset_config,
            max_samples=max_samples,
            seed=seed,
        )
    else:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, dataset_config, split=split)
        _validate_max_samples(max_samples, len(ds))
        if max_samples and len(ds) > max_samples:
            import random
            rng = random.Random(seed)
            idx = rng.sample(range(len(ds)), max_samples)
            ds = ds.select(idx)

        tokenizer = get_tokenizer(model_name)
        _require_keys(data_cfg, "data", ["text_col", "label_col"])
        text_col = data_cfg["text_col"]
        label_col = data_cfg["label_col"]
        if text_col not in ds.column_names:
            raise KeyError(
                f"text_col '{text_col}' not found in dataset columns: {ds.column_names}"
            )
        if label_col not in ds.column_names:
            raise KeyError(
                f"label_col '{label_col}' not found in dataset columns: {ds.column_names}"
            )

        def collate(examples):
            batch = tokenizer(
                [ex[text_col] for ex in examples],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch["labels"] = torch.tensor([ex[label_col] for ex in examples])
            return dict(batch)

    loader = DataLoader(
        ds,
        batch_size=train_cfg["batch_size"],
        shuffle=(split == "train"),
        collate_fn=collate,
    )
    return loader, collate
