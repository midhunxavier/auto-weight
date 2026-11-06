"""Evaluate a serialized Qwen model with fixed seed/batch units and optional local resume."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

from math_verify import parse, verify
import torch
from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList, Qwen3_5ForConditionalGeneration


class PresencePenalty(LogitsProcessor):
    def __init__(self, start):
        self.start = start

    def __call__(self, ids, scores):
        for index, sequence in enumerate(ids):
            scores[index, sequence[self.start:].unique()] -= 1.5
        return scores


def correct(text, answer):
    marker = text.rfind(r"\boxed{")
    if marker < 0:
        return False
    depth = 1
    for end in range(marker + len(r"\boxed{"), len(text)):
        depth += (text[end] == "{") - (text[end] == "}")
        if depth == 0:
            try:
                return bool(verify(parse(r"\boxed{" + str(answer) + "}"), parse(text[marker:end + 1])))
            except (TypeError, ValueError):
                return False
    return False


def run_batch(model, tokenizer, rows, device, seed, cap):
    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": row["inputs"]["problem"] +
          "\nPlease reason step by step, and put your final answer within \\boxed{}."}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True) for row in rows]
    tokens = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    width = tokens.input_ids.shape[1]
    torch.manual_seed(seed)
    with torch.inference_mode():
        result = model.generate(**tokens, max_new_tokens=cap, do_sample=True, temperature=0.6,
                                top_p=0.95, top_k=20, logits_processor=LogitsProcessorList([PresencePenalty(width)]),
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    for row, sequence in zip(rows, result[:, width:].tolist(), strict=True):
        eos = tokenizer.eos_token_id
        if eos in sequence:
            sequence = sequence[:sequence.index(eos) + 1]
        text = tokenizer.decode(sequence, skip_special_tokens=False)
        yield {"id": row["id"], "metrics": {
            "accuracy": int(correct(text, row["inputs"]["answer"])),
            "cap_hit": int(len(sequence) == cap and sequence[-1] != eos),
            "tokens": len(sequence),
        }}


def load_unit(path, identity, ids, seed, cap):
    if path is None or not path.is_file():
        return None
    unit = json.loads(path.read_text())
    rows = unit["observations"]
    if unit["identity"] != identity or [row["id"] for row in rows] != ids:
        raise ValueError("cached unit identity differs from request")
    for row in rows:
        metrics = row["metrics"]
        if (row["seed"] != seed or set(metrics) != {"accuracy", "cap_hit", "tokens"}
                or metrics["accuracy"] not in (0, 1) or metrics["cap_hit"] not in (0, 1)
                or type(metrics["tokens"]) is not int or not 1 <= metrics["tokens"] <= cap):
            raise ValueError("invalid cached Qwen observation")
    return rows


def main(request_path, response_path):
    request = json.loads(Path(request_path).read_text())
    rows = [json.loads(line) for line in Path(request["cases"]).read_text().splitlines()]
    config = request["config"]
    device, batch, cap = config["device"], config["batch_size"], config["max_tokens"]
    if device not in ("cpu", "cuda") or not 1 <= batch <= 64 or not 1 <= cap <= 32768:
        raise ValueError("invalid Qwen evaluation limits")
    if len(rows) * len(request["seeds"]) * cap > request["max_work"]:
        raise ValueError("maximum possible generation exceeds declared token budget")
    torch.set_num_threads(4)
    environment = [sys.version, str(torch.__version__), importlib.metadata.version("transformers"),
                   importlib.metadata.version("math-verify"), torch.cuda.get_device_name() if device == "cuda" else "cpu"]
    cache_root = os.environ.get("AUTOWEIGHT_QWEN_CACHE")
    cache = None if cache_root is None else Path(cache_root).resolve()
    if cache is not None:
        cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    model = tokenizer = None
    observations = []
    for seed in request["seeds"]:
        for start in range(0, len(rows), batch):
            subset = rows[start:start + batch]
            ids = [row["id"] for row in subset]
            identity = hashlib.sha256(json.dumps([request["evaluation_id"], environment, seed, start, ids],
                                                  sort_keys=True).encode()).hexdigest()
            path = cache / f"{identity}.json" if cache is not None else None
            result = load_unit(path, identity, ids, seed, cap)
            if result is None:
                if model is None:
                    tokenizer = AutoTokenizer.from_pretrained(request["artifact"], padding_side="left",
                                                              local_files_only=True, trust_remote_code=False)
                    model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
                        request["artifact"], dtype=torch.float32, attn_implementation="sdpa",
                        local_files_only=True, use_safetensors=True, output_loading_info=True)
                    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
                        raise ValueError("Qwen checkpoint did not load completely")
                    model = model.eval().to(device)
                result = [{**item, "seed": seed} for item in run_batch(model, tokenizer, subset, device, seed + start, cap)]
                if path is not None:
                    with NamedTemporaryFile(mode="w", dir=cache, delete=False) as temporary:
                        json.dump({"identity": identity, "observations": result}, temporary)
                    Path(temporary.name).replace(path)
            observations.extend(result)
    work = sum(row["metrics"]["tokens"] for row in observations)
    Path(response_path).write_text(json.dumps({
        **{key: request[key] for key in ("version", "evaluation_id", "artifact_sha256")},
        "observations": observations, "work": work,
    }))


if __name__ == "__main__":
    main(*sys.argv[1:])
