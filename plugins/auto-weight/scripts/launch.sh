#!/bin/sh
set -eu

config="${AUTOWEIGHT_CONFIG:-${XDG_CONFIG_HOME:-$HOME/.config}/auto-weight/editor.conf}"
runtime=""
input_root=""
runs_dir=""
if [ -f "$config" ]; then
    {
        IFS= read -r runtime || :
        IFS= read -r input_root || :
        IFS= read -r runs_dir || :
    } < "$config"
fi
runtime="${AUTOWEIGHT_PYTHON:-${runtime:-$HOME/.local/share/auto-weight/venv/bin/python}}"
input_root="${AUTOWEIGHT_INPUT_ROOT:-$input_root}"
runs_dir="${AUTOWEIGHT_RUNS_DIR:-${runs_dir:-$HOME/.local/share/auto-weight/runs}}"

if [ ! -x "$runtime" ] || [ -z "$input_root" ]; then
    printf '%s\n' "Run auto-weight setup-runtime --input-root /path/to/repair-inputs to configure $config." >&2
    exit 1
fi

exec "$runtime" -m auto_weight.mcp_server --input-root "$input_root" --runs-dir "$runs_dir"
