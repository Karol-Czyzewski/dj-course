"""Generuje jeden plik PNG przedstawiający kształty wybranych tensorów dla modeli.

Uruchomienie:
    python aa.py
Wynik:
    figures/tensor_shapes.png (tabela z adnotacjami oraz heatmapa liczby parametrów)
"""

from pathlib import Path
import json
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CONFIG_DIR = Path(__file__).parent / "hf-configs"
OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

tensors = [
    "embed_tokens.weight",
    "input_layernorm.weight",
    "mlp.down_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "post_attention_layernorm.weight",
    "self_attn.k_proj.weight",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
]

def tensor_shape(json_data: dict, name: str):
    hs = json_data.get("hidden_size", 0)
    vocab = json_data.get("vocab_size", 0)
    inter = json_data.get("intermediate_size", 0)
    n_heads = json_data.get("num_attention_heads", 1) or 1
    if name == "embed_tokens.weight":
        return [vocab, hs]
    if name == "input_layernorm.weight":
        return [hs]
    if name == "mlp.down_proj.weight":
        return [inter, hs]
    if name == "mlp.gate_proj.weight":
        return [inter, hs]
    if name == "mlp.up_proj.weight":
        return [hs, inter]
    if name == "post_attention_layernorm.weight":
        return [hs]
    if name in {"self_attn.k_proj.weight", "self_attn.q_proj.weight", "self_attn.v_proj.weight"}:
        return [hs, hs // n_heads]
    if name == "self_attn.o_proj.weight":
        return [hs, hs]
    return [math.nan, math.nan]

def shape_to_str(shape):
    return "x".join(str(s) for s in shape if not (isinstance(s, float) and math.isnan(s)))

def param_count(shape):
    try:
        total = 1
        for s in shape:
            total *= int(s)
        return total
    except Exception:
        return math.nan

def load_configs():
    rows = []
    for p in sorted(CONFIG_DIR.glob("*.json")):
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        rows.append({"filename": p.name, "json": data})
    return pd.DataFrame(rows)

def build_tensor_df(df_models: pd.DataFrame) -> pd.DataFrame:
    table = {}
    for _, row in df_models.iterrows():
        fname = row["filename"].replace("-config.json", "")
        js = row["json"]
        shapes = [tensor_shape(js, t) for t in tensors]
        table[fname] = [shape_to_str(s) for s in shapes]
    return pd.DataFrame(table, index=tensors)

def build_param_matrix(df_models: pd.DataFrame) -> pd.DataFrame:
    matrix = {}
    for _, row in df_models.iterrows():
        fname = row["filename"].replace("-config.json", "")
        js = row["json"]
        counts = [param_count(tensor_shape(js, t)) for t in tensors]
        matrix[fname] = counts
    return pd.DataFrame(matrix, index=tensors)

def plot_tensor_table(df_shapes: pd.DataFrame, df_params: pd.DataFrame):
    # Normalizacja do log10 dla heatmapy parametrów (pomija NaN)
    data = df_params.astype(float).copy()
    with np.errstate(invalid="ignore"):
        data = np.log10(data)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(data, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(df_shapes.columns)))
    ax.set_xticklabels(df_shapes.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(df_shapes.index)))
    ax.set_yticklabels(df_shapes.index)
    ax.set_title("Tensor shapes (annotacje) + heatmap log10(param_count)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10(param_count)")

    # Adnotacje: shape string
    for i in range(len(df_shapes.index)):
        for j in range(len(df_shapes.columns)):
            shape_txt = df_shapes.iloc[i, j]
            ax.text(j, i, shape_txt, ha="center", va="center", color="white", fontsize=8)

    plt.tight_layout()
    out = OUT_DIR / "tensor_shapes.png"
    plt.savefig(out, dpi=150)
    print(f"Zapisano: {out}")
    plt.close()

def main():
    if not CONFIG_DIR.exists():
        raise SystemExit(f"Brak katalogu z konfiguracjami: {CONFIG_DIR}")
    df_models = load_configs()
    if df_models.empty:
        raise SystemExit("Nie znaleziono plików konfiguracyjnych")
    df_shapes = build_tensor_df(df_models)
    df_params = build_param_matrix(df_models)
    print("Tabela kształtów:")
    print(df_shapes)
    plot_tensor_table(df_shapes, df_params)

if __name__ == "__main__":
    main()