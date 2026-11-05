"""Bounded greedy generation and deterministic grading of entire text completions."""
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from referencing import Registry
from transformers import GenerationConfig

from .contracts import EditError, canonical, fields, integer, strict_json, string


def compile_check(check):
    if not isinstance(check, dict):
        raise EditError("text check must be an object")
    if check.get("type") == "exact":
        fields(check, {"type", "expected"}, "exact check")
        expected = string(check["expected"], "expected text")
        return ("text_exact",), lambda text: (text.strip() == expected.strip(),)
    if check.get("type") != "json":
        raise EditError("unsupported text check type")
    if check.keys() - {"type", "schema", "expected"}:
        raise EditError("unknown JSON check field")
    validator = None
    metrics = ["json_valid"]
    if "schema" in check:
        schema = check["schema"]
        # The supported dialect is fixed even if a schema declares a different one.
        if isinstance(schema, dict) and schema.get("$schema", "https://json-schema.org/draft/2020-12/schema")                 not in ("https://json-schema.org/draft/2020-12/schema", "https://json-schema.org/draft/2020-12/schema#"):
            raise EditError("JSON checks support Draft 2020-12 schemas")
        try:
            Draft202012Validator.check_schema(schema)
            validator = Draft202012Validator(schema, registry=Registry())
            if "expected" in check and not validator.is_valid(check["expected"]):
                raise EditError("expected JSON does not match its schema")
        except SchemaError as exc:
            raise EditError(f"invalid JSON Schema: {exc.message}") from exc
        metrics.append("schema_match")
    expected = canonical(check["expected"]) if "expected" in check else None
    if expected is not None:
        metrics.append("json_exact")

    def grade(text):
        try:
            parsed = strict_json(text)
        except EditError:
            return (False,) * len(metrics)
        scores = [True]
        if validator is not None:
            scores.append(validator.is_valid(parsed))
        if expected is not None:
            scores.append(canonical(parsed) == expected)
        return tuple(scores)

    return tuple(metrics), grade


def read_examples(path, tokenizer, config, gates, model_context):
    groups = defaultdict(list)
    group_metrics = {}
    seen = set()
    with Path(path).open("rb") as stream:
        for index, raw in enumerate(stream, 1):
            integer(index, "text example count", 1, 10000)
            row = strict_json(raw, 32 * 1024)
            fields(row, {"group", "prompt", "check"}, f"example {index}")
            group = string(row["group"], "group", 80)
            prompt = string(row["prompt"], "prompt", 12000)
            metrics, grade = compile_check(row["check"])
            if group_metrics.setdefault(group, metrics) != metrics:
                raise EditError(f"examples in {group} must use the same metrics")
            if len(group_metrics) > 16:
                raise EditError("text evaluation exceeds sixteen groups")
            chat = config["prompt_format"] == "chat"
            rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                      add_generation_prompt=True) if chat else prompt
            tokens = tuple(tokenizer.encode(rendered, add_special_tokens=not chat))
            if not 0 < len(tokens) <= config["max_input_tokens"] or len(tokens) + config["max_new_tokens"] > model_context:
                raise EditError(f"example {index} exceeds the configured input or model context limit")
            if tokens in seen:
                raise EditError("duplicate tokenized evaluation prompt")
            seen.add(tokens)
            groups[group].append((prompt, tokens, grade))
    for gate in gates:
        group = gate["group"]
        if len(groups.get(group, [])) < 32 or gate["metric"] not in group_metrics[group]:
            raise EditError(f"gate {group}/{gate['metric']} requires at least 32 examples with that metric")
    return groups, group_metrics, seen


def generation_policy(model, tokenizer, max_new_tokens):
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = eos[0] if isinstance(eos, list) and eos else eos
    if pad is None:
        raise EditError("generation requires a pad or EOS token")
    return GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                            pad_token_id=pad, eos_token_id=eos, use_cache=True)


@torch.inference_mode()
def score(model, tokenizer, groups, batch_size, policy):
    scores, samples = {}, {}
    for group, examples in groups.items():
        rows, completions = [], []
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            length = max(len(tokens) for _, tokens, _ in batch)
            inputs = torch.full((len(batch), length), policy.pad_token_id, dtype=torch.long)
            attention = torch.zeros_like(inputs)
            for index, (_, tokens, _) in enumerate(batch):
                inputs[index, -len(tokens):] = torch.tensor(tokens)
                attention[index, -len(tokens):] = 1
            output = model.generate(input_ids=inputs, attention_mask=attention, generation_config=policy)
            texts = tokenizer.batch_decode(output[:, length:], skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)
            rows.extend(grade(text) for (_, _, grade), text in zip(batch, texts, strict=True))
            if len(completions) < 2:
                completions.extend(text[:400] for text in texts[:2 - len(completions)])
        scores[group] = np.asarray(rows, dtype="float64")
        samples[group] = completions
    return scores, samples
