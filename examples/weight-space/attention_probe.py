"""Compare attention to selected prompt tokens in two compatible local checkpoints."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch

from auto_weight.checkpoint import HF_ADAPTER, checkpoint_hash
from auto_weight.contracts import EditError
from auto_weight.hf import check_donor_metadata, load_model, load_tokenizer


def probe(paths, prompt, start, stop, query):
    check_donor_metadata(*paths)
    tokenizer = load_tokenizer(paths[0], {"prompt_format": "raw"})
    tokens = tokenizer(prompt, return_tensors="pt")
    count = tokens.input_ids.shape[1]
    if query == -1:
        query = count - 1
    if not 1 <= count <= 128 or not 0 <= start < stop <= query + 1 <= count:
        raise EditError("use 1–128 prompt tokens and a source span at or before the query token")
    reports = {}
    for label, path in zip(("A", "B"), paths, strict=True):
        model = load_model(path)
        with torch.inference_mode():
            output = model(**tokens, output_attentions=True, use_cache=False)
        if not output.attentions or any(layer is None for layer in output.attentions):
            raise EditError("checkpoint does not expose eager attention")
        masses = [layer[0, :, query, start:stop].float().sum(-1) for layer in output.attentions]
        if any(not torch.isfinite(layer).all() for layer in masses):
            raise EditError("attention contains nonfinite values")
        reports[label] = {"sha256": checkpoint_hash(path, HF_ADAPTER),
                          "attention_mass_by_layer_and_head": [layer.tolist() for layer in masses]}
        del output, model
    return {"prompt_sha256": sha256(prompt.encode()).hexdigest(), "query": query,
            "source_span": [start, stop], "prompt_tokens": count, "checkpoints": reports}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--source", type=int, nargs=2, required=True, metavar=("START", "STOP"))
    parser.add_argument("--query", type=int, default=-1, help="query token index (default: last token)")
    args = parser.parse_args()
    try:
        print(json.dumps(probe((args.first, args.second), args.prompt, *args.source, args.query), indent=2))
    except (EditError, OSError) as exc:
        parser.error(str(exc))
