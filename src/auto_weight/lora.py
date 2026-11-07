"""Load one standard PEFT LoRA adapter on a pinned local backbone."""
from pathlib import Path
import re
import shutil

import torch
from peft import PeftConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors import SafetensorError
from safetensors.torch import load_file, save_file

from .checkpoint import LORA_CONFIG, LORA_WEIGHTS
from .contracts import EditError, read_json

LORA_TENSOR = re.compile(r"base_model\.model\..+\.(lora_[AB]\.weight|lora_embedding_[AB])")


def load_config(path):
    data = read_json(path / LORA_CONFIG)
    if not isinstance(data, dict) or data.get("peft_type") != "LORA":
        raise EditError("expected a standard PEFT LoRA adapter")
    unsupported = (
        "modules_to_save", "lora_bias", "use_rslora", "use_dora", "use_qalora", "alora_invocation_tokens",
        "layer_replication", "trainable_token_indices", "target_parameters", "megatron_config", "arrow_config",
    )
    if (any(data.get(name) for name in unsupported)
            or data.get("bias", "none") != "none"
            or data.get("task_type") not in (None, "CAUSAL_LM")
            or data.get("init_lora_weights", True) not in (True, False, "gaussian")):
        raise EditError("LoRA repair requires a standard causal LM adapter with only LoRA A and B weights")
    try:
        config = PeftConfig.from_pretrained(path, local_files_only=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise EditError(f"invalid PEFT LoRA config: {exc}") from exc
    config.inference_mode = True
    return config


def load_weights(path):
    try:
        weights = load_file(path / LORA_WEIGHTS)
    except SafetensorError as exc:
        raise EditError(f"invalid LoRA safetensors: {exc}") from exc
    if not weights or any(
        not LORA_TENSOR.fullmatch(name)
        or tensor.ndim != 2 or tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        or not torch.isfinite(tensor).all()
        for name, tensor in weights.items()
    ):
        raise EditError("LoRA safetensors must contain only finite floating-point LoRA A and B matrices")
    return weights


def attach(model, path):
    config = load_config(path)
    weights = load_weights(path)
    try:
        adapted = get_peft_model(model, config)
    except (TypeError, ValueError) as exc:
        raise EditError(f"LoRA configuration does not match the supplied base model: {exc}") from exc
    expected = get_peft_model_state_dict(adapted, save_embedding_layers=False)
    if expected.keys() != weights.keys() or any(expected[key].shape != value.shape for key, value in weights.items()):
        raise EditError("LoRA tensor names or shapes do not match the supplied base model and adapter config")
    result = set_peft_model_state_dict(adapted, weights)
    if result.unexpected_keys:
        raise EditError("PEFT could not load every LoRA tensor")
    return adapted.eval()


def save_candidate(source: Path, destination: Path, weights):
    destination.mkdir()
    shutil.copyfile(source / LORA_CONFIG, destination / LORA_CONFIG)
    save_file(weights, destination / LORA_WEIGHTS, metadata={"format": "pt"})
