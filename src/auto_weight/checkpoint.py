"""Content identities for the local checkpoint formats understood by the runner."""
from pathlib import Path
import re

from .contracts import EditError, file_hash, identify, read_json

HF_ADAPTER = "hf-causal-lm-v1"
LORA_ADAPTER = "hf-lora-v1"
RAW_HF_ADAPTER = "hf-safetensors-v1"
NAMED_ADAPTER = "named-tensors-v1"


def hf_storage(adapter):
    return adapter in {HF_ADAPTER, RAW_HF_ADAPTER, LORA_ADAPTER}


def hf_text(adapter):
    return adapter in {HF_ADAPTER, LORA_ADAPTER}

HF_LIMIT = 4 * 1024**3
HF_METADATA = {
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
    "vocab.json", "merges.txt", "tokenizer.model", "spiece.model", "vocab.txt",
    "preprocessor_config.json", "video_preprocessor_config.json",
}
REQUIRED = {"config.json", "tokenizer.json", "tokenizer_config.json"}
WEIGHT_FILE = "model.safetensors"
INDEX_FILE = "model.safetensors.index.json"
LORA_CONFIG = "adapter_config.json"
LORA_WEIGHTS = "adapter_model.safetensors"
SHARD = re.compile(r"model(?:\.safetensors)?-[0-9]{5}-of-[0-9]{5}\.safetensors")


def artifact_name(adapter):
    if adapter == LORA_ADAPTER:
        return "adapter"
    return "model" if hf_storage(adapter) else "model.npz"


def directory_adapter(path):
    path = Path(path)
    if (path / LORA_CONFIG).is_file() and not any((path / name).exists() for name in (WEIGHT_FILE, INDEX_FILE)):
        return LORA_ADAPTER
    return HF_ADAPTER


def checkpoint_files(path, adapter):
    if adapter != LORA_ADAPTER:
        return hf_files(path)
    path = Path(path)
    files = {name: path / name for name in (LORA_CONFIG, LORA_WEIGHTS)}
    if not all(file.is_file() for file in files.values()):
        raise EditError("LoRA adapter needs adapter_config.json and adapter_model.safetensors")
    if any((path / name).exists() for name in (WEIGHT_FILE, INDEX_FILE)):
        raise EditError("supply a standalone PEFT LoRA adapter folder")
    if files[LORA_CONFIG].stat().st_size > 1024**2 or files[LORA_WEIGHTS].stat().st_size > HF_LIMIT:
        raise EditError("LoRA adapter exceeds the file size limit")
    return files


def hf_files(path):
    """Only copy assets used by the supported offline Transformers loader."""
    path = Path(path)
    if not path.is_dir():
        raise EditError(f"expected a Hugging Face checkpoint directory: {path}")
    files = {}
    for entry in path.iterdir():
        if entry.name in HF_METADATA | {WEIGHT_FILE, INDEX_FILE} or SHARD.fullmatch(entry.name):
            if not entry.is_file():
                raise EditError(f"invalid checkpoint file: {entry.name}")
            files[entry.name] = entry
    names = files.keys()
    shards = {name for name in names if SHARD.fullmatch(name)}
    if WEIGHT_FILE not in names and INDEX_FILE not in names and (path / LORA_CONFIG).is_file():
        raise EditError("pass the original model as --base and the standalone PEFT adapter as --lora")
    if not REQUIRED <= names:
        raise EditError("checkpoint needs config.json, tokenizer.json and tokenizer_config.json")
    if WEIGHT_FILE in names:
        if INDEX_FILE in names or shards:
            raise EditError("checkpoint mixes unsharded and sharded weights")
    elif INDEX_FILE in names and shards:
        index = read_json(files[INDEX_FILE], 16 * 1024**2)
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if (not isinstance(weight_map, dict) or not weight_map
                or any(not isinstance(value, str) for value in weight_map.values())
                or set(weight_map.values()) != shards):
            raise EditError("safetensors index must name only the checkpoint's local shards")
    else:
        raise EditError("checkpoint needs model.safetensors or a complete safetensors shard index")
    if len(files) > 64 or sum(file.stat().st_size for file in files.values()) > HF_LIMIT:
        raise EditError("Hugging Face checkpoint exceeds the file or byte limit")
    return files


def checkpoint_hash(path, adapter, *, frozen=False):
    if not hf_storage(adapter):
        return file_hash(path)
    files = checkpoint_files(path, adapter)
    if frozen and {entry.name for entry in Path(path).iterdir()} != files.keys():
        raise EditError("unexpected files in frozen Hugging Face checkpoint")
    return identify({name: file_hash(file) for name, file in files.items()})


def tensor_metadata(path, prefix, limit):
    """Read names, shapes and dtypes from safetensors headers."""
    try:
        from safetensors import SafetensorError, safe_open
    except ImportError as exc:
        raise EditError("install auto-weight[hf] to list Hugging Face tensors") from exc

    tensors = {}
    try:
        for name, file in checkpoint_files(path, directory_adapter(path)).items():
            if not name.endswith(".safetensors"):
                continue
            with safe_open(file, framework="np") as weights:
                for key in weights.keys():
                    if not key.startswith(prefix):
                        continue
                    if key in tensors:
                        raise EditError(f"duplicate stored tensor: {key}")
                    view = weights.get_slice(key)
                    tensors[key] = {"name": key, "shape": view.get_shape(), "dtype": view.get_dtype()}
    except SafetensorError as exc:
        raise EditError(f"invalid safetensors checkpoint: {exc}") from exc
    names = sorted(tensors)
    return {"status": "listed", "matching": len(names), "truncated": len(names) > limit,
            "tensors": [tensors[name] for name in names[:limit]]}
