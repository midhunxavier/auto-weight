"""Offline adapter for standard Hugging Face causal language model checkpoints."""
from pathlib import Path
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .checkpoint import HF_ADAPTER, HF_METADATA, checkpoint_hash, hf_files
from .contracts import EditError
from .evaluate import paired_report
from . import text_eval


def load_model(path):
    model, info = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        dtype="auto", attn_implementation="eager", output_loading_info=True,
    )
    for field in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if info.get(field):
            raise EditError(f"checkpoint does not match its model config: {field}")
    if any(param.device.type != "cpu" or param.dtype not in (torch.float32, torch.float16, torch.bfloat16)
           for param in model.parameters()):
        raise EditError("adapter requires a dense CPU model with float32, float16 or bfloat16 parameters")
    return model.eval()


def model_config(model):
    config = model.config.to_dict()
    for key in ("_name_or_path", "transformers_version"):
        config.pop(key, None)
    return config


def check_tokenizer_match(base_path, donor_path):
    names = HF_METADATA - {"config.json", "generation_config.json"}
    from .contracts import file_hash

    def identity(path):
        return {name: file_hash(path / name) for name in names if (path / name).exists()}

    if identity(base_path) != identity(donor_path):
        raise EditError("donor tokenizer assets differ from base")


@torch.no_grad()
def apply(model, base_path, proposal, proposal_dir):
    parameters = dict(model.named_parameters(remove_duplicate=False))
    addressed = set()
    changes = []
    for operation in proposal["operations"]:
        names, kind = operation["tensors"], operation["op"]
        for name in names:
            if name not in parameters:
                raise EditError(f"unknown parameter: {name}")
            address = id(parameters[name])
            if address in addressed:
                raise EditError(f"tied parameter addressed more than once: {name}")
            addressed.add(address)
        if kind != "scale":
            donor_path = proposal_dir / operation["donor"]["path"]
            check_tokenizer_match(base_path, donor_path)
            donor = load_model(donor_path)
            if model_config(donor) != model_config(model):
                raise EditError("donor architecture config differs from base")
            donor_parameters = dict(donor.named_parameters(remove_duplicate=False))
        for name in names:
            parameter = parameters[name]
            before = parameter.double()
            if kind == "scale":
                after = (before * operation["factor"]).to(parameter.dtype)
            else:
                source = donor_parameters.get(name)
                if source is None or source.shape != parameter.shape or source.dtype != parameter.dtype:
                    raise EditError(f"donor parameter shape or dtype differs: {name}")
                if kind == "restore":
                    after = source
                else:
                    alpha = operation["alpha"]
                    after = ((1 - alpha) * before + alpha * source.double()).to(parameter.dtype)
            if not torch.isfinite(after).all():
                raise EditError(f"nonfinite edited parameter: {name}")
            delta = after.double() - before
            count = int(torch.count_nonzero(delta))
            if count:
                changes.append({"tensor": name, "op": kind, "parameters_changed": count,
                                "l2_delta": delta.norm().item(), "max_abs_delta": delta.abs().max().item()})
                parameter.copy_(after)
        if kind != "scale":
            del donor_parameters, donor
    if not changes:
        raise EditError("proposal changes no parameter values")
    return changes


def save_candidate(model, base_path, destination):
    model.save_pretrained(destination, safe_serialization=True, max_shard_size="1GB")
    # Keep the complete original inference metadata byte-for-byte.
    for name in HF_METADATA:
        source, target = base_path / name, destination / name
        if source.is_file():
            shutil.copyfile(source, target)
        elif target.exists():
            target.unlink()
    hf_files(destination)


def evaluate(workspace, issue, proposal, proposal_dir, stage, temporary, expected_candidate=None):
    torch.set_num_threads(issue["budget"]["threads"])
    torch.set_num_interop_threads(1)
    base_path = workspace / issue["base"]["path"]
    base = load_model(base_path)
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True, trust_remote_code=False,
                                               use_fast=True)
    if not tokenizer.is_fast:
        raise EditError("adapter requires a fast tokenizer")
    config, gates = issue["model"]["config"], issue["gates"]
    batch = issue["budget"]["batch_size"]
    if batch * config["max_input_tokens"] * base.config.vocab_size > 64_000_000:
        raise EditError("text batch exceeds the evaluator logits budget")
    context = getattr(base.config, "max_position_embeddings", config["max_input_tokens"] + config["max_new_tokens"])
    if not isinstance(context, int):
        context = config["max_input_tokens"] + config["max_new_tokens"]
    policy = text_eval.generation_policy(base, tokenizer, config["max_new_tokens"])
    changes = apply(base, base_path, proposal, proposal_dir)
    candidate_path = temporary / "model"
    save_candidate(base, base_path, candidate_path)
    del base
    candidate_sha = checkpoint_hash(candidate_path, HF_ADAPTER, frozen=True)
    if expected_candidate is not None and candidate_sha != expected_candidate:
        raise EditError("replayed candidate differs from development candidate")
    groups, metrics, prompts = text_eval.read_examples(
        workspace / issue["datasets"][stage]["path"], tokenizer, config, gates, context)
    if stage == "confirmation":
        _, _, development_prompts = text_eval.read_examples(
            workspace / issue["datasets"]["development"]["path"], tokenizer, config, gates, context)
        if development_prompts & prompts:
            raise EditError("development and confirmation prompts overlap after tokenization")
    before, before_samples = text_eval.score(load_model(base_path), tokenizer, groups, batch, policy)
    after, after_samples = text_eval.score(load_model(candidate_path), tokenizer, groups, batch, policy)
    report = paired_report(before, after, gates, metrics)
    report["inference"]["generation"] = {"strategy": "greedy", **config,
                                         "eos_token_id": policy.eos_token_id, "pad_token_id": policy.pad_token_id}
    if stage == "development":
        report["development_samples"] = {
            group: [{"prompt": prompt[:400], "baseline": original, "candidate": edited}
                    for (prompt, _, _), original, edited
                    in zip(rows, before_samples[group], after_samples[group])]
            for group, rows in groups.items()
        }
    return {**report, "changes": changes, "candidate_sha256": candidate_sha}
