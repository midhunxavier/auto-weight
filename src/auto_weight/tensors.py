"""Bounded checkpoint inspection and data-only replacement patches."""
from math import prod
from zipfile import BadZipFile, ZipFile

import numpy as np

from .checkpoint import checkpoint_files, checkpoint_hash, directory_adapter, tensor_metadata
from .contracts import CHECKPOINT_LIMIT, EditError, canonical, integer

PATCH_LIMIT = 64 * 1024**2
MAX_CHANGED = 12_000_000
READ_LIMIT = 4096


def npz_layout(path, limit=CHECKPOINT_LIMIT):
    layout = {}
    try:
        with ZipFile(path) as archive:
            entries = archive.infolist()
            if not 1 <= len(entries) <= 10000 or sum(item.file_size for item in entries) > limit:
                raise EditError("NPZ exceeds the tensor or expanded byte limit")
            for entry in entries:
                name = entry.filename.removesuffix(".npy")
                if name == entry.filename or not name or name in layout:
                    raise EditError("invalid or duplicate NPZ tensor name")
                with archive.open(entry) as stream:
                    version = np.lib.format.read_magic(stream)
                    readers = {(1, 0): np.lib.format.read_array_header_1_0,
                               (2, 0): np.lib.format.read_array_header_2_0}
                    if version not in readers:
                        raise EditError("unsupported NPY version")
                    shape, _, dtype = readers[version](stream)
                    if dtype.kind not in "fiu" or dtype.itemsize > 8 or prod(shape) * dtype.itemsize != entry.file_size - stream.tell():
                        raise EditError("NPZ requires bounded, real numeric tensors")
                layout[name] = (shape, dtype)
    except (BadZipFile, ValueError, EOFError) as exc:
        raise EditError(f"invalid NPZ: {exc}") from exc
    return layout


def load_arrays(path, limit=CHECKPOINT_LIMIT):
    layout = npz_layout(path, limit)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in layout}
    if any(not np.isfinite(array).all() for array in arrays.values()):
        raise EditError("NPZ contains nonfinite values")
    return arrays


def region(shape, bounds):
    if bounds is None:
        return tuple(slice(None) for _ in shape), tuple(shape)
    if not isinstance(bounds, list) or len(bounds) != len(shape):
        raise EditError("slice must contain one [start, stop] pair per axis")
    indexes, lengths = [], []
    for size, pair in zip(shape, bounds, strict=True):
        if not isinstance(pair, list) or len(pair) != 2:
            raise EditError("slice requires [start, stop] pairs")
        start = integer(pair[0], "slice start", 0, size - 1)
        stop = integer(pair[1], "slice stop", start + 1, size)
        indexes.append(slice(start, stop))
        lengths.append(stop - start)
    return tuple(indexes), tuple(lengths)


def metadata(path, prefix, limit):
    if path.is_dir():
        return tensor_metadata(path, prefix, limit)
    layout = npz_layout(path)
    names = sorted(name for name in layout if name.startswith(prefix))
    return {"status": "listed", "matching": len(names), "truncated": len(names) > limit,
            "tensors": [{"name": name, "shape": list(layout[name][0]), "dtype": str(layout[name][1])}
                        for name in names[:limit]]}


def read_tensor(path, name, bounds=None, summary=False):
    adapter = directory_adapter(path) if path.is_dir() else None
    identity = checkpoint_hash(path, adapter)
    if path.is_dir():
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise EditError("install auto-weight[hf] to read Hugging Face tensors") from exc
        for filename, file in checkpoint_files(path, adapter).items():
            if filename.endswith(".safetensors"):
                with safe_open(file, framework="pt") as archive:
                    if name in archive.keys():
                        view = archive.get_slice(name)
                        indexes, shape = region(view.get_shape(), bounds)
                        integer(prod(shape), "slice element count", 1, READ_LIMIT)
                        values = view[indexes].double().numpy()
                        break
        else:
            raise EditError("unknown tensor")
    else:
        layout = npz_layout(path)
        if name not in layout:
            raise EditError("unknown tensor")
        indexes, shape = region(layout[name][0], bounds)
        integer(prod(shape), "slice element count", 1, READ_LIMIT)
        with np.load(path, allow_pickle=False) as archive:
            values = archive[name][indexes].astype("float64")
    if not np.isfinite(values).all():
        raise EditError("tensor slice contains nonfinite values")
    if checkpoint_hash(path, adapter) != identity:
        raise EditError("checkpoint changed while reading")
    result = {"status": "read", "sha256": identity, "tensor": name, "shape": list(shape)}
    if summary:
        return {**result, "min": float(values.min()), "max": float(values.max()),
                "mean": float((values / values.size).sum())}
    payload = values.tolist()
    if len(canonical(payload)) > 64 * 1024:
        raise EditError("tensor response exceeds 64 KiB; request a smaller slice")
    return {**result, "values": payload}


def replacement_items(operation, directory):
    payload = load_arrays(directory / operation["payload"]["path"], PATCH_LIMIT)
    if set(payload) != set(operation["tensors"]):
        raise EditError("replacement payload must contain exactly the declared tensors")
    for name in operation["tensors"]:
        yield name, operation["slices"].get(name), payload[name]


def replacement_slice(name, target, bounds, values):
    indexes, shape = region(target.shape, bounds)
    if values.shape != shape:
        raise EditError(f"replacement shape differs for {name}")
    return indexes


def replace_torch(name, target, bounds, values):
    import torch

    if not target.is_floating_point():
        raise EditError(f"replacement requires a floating point tensor: {name}")
    before = target[replacement_slice(name, target, bounds, values)]
    after = torch.from_numpy(values.astype("float64")).to(before.dtype)
    if not torch.isfinite(after).all():
        raise EditError(f"nonfinite replacement for {name}")
    item = change(name, before.double().numpy(), after.double().numpy())
    if item:
        before.copy_(after)
    return item


def change(name, before, after):
    delta = after.astype("float64") - before.astype("float64")
    count = int(np.count_nonzero(before != after))
    if count:
        norm, maximum = float(np.linalg.norm(delta.ravel())), float(np.abs(delta).max())
        if not np.isfinite([norm, maximum]).all():
            raise EditError("replacement delta exceeds the numeric limit")
        return {"tensor": name, "op": "replace", "parameters_changed": count,
                "l2_delta": norm, "max_abs_delta": maximum}


def check_changes(changes):
    if not changes:
        raise EditError("proposal changes no tensor values")
    if sum(item["parameters_changed"] for item in changes if item["op"] == "replace") > MAX_CHANGED:
        raise EditError("replacement exceeds the changed parameter limit")


def replace_arrays(base, candidate, operation, directory):
    changes = []
    for name, bounds, values in replacement_items(operation, directory):
        if name not in base:
            raise EditError(f"unknown tensor: {name}")
        indexes = replacement_slice(name, base[name], bounds, values)
        with np.errstate(over="ignore", invalid="ignore"):
            after = values.astype(base[name].dtype)
        if not np.isfinite(after).all():
            raise EditError(f"nonfinite replacement for {name}")
        if after.dtype.kind in "iu" and not np.array_equal(values, after):
            raise EditError(f"replacement cannot be represented in the integer tensor {name}")
        item = change(name, base[name][indexes], after)
        if item:
            changes.append(item)
            candidate[name] = candidate[name].copy()
            candidate[name][indexes] = after
    return changes
