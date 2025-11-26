"""Prosta wizualizacja porównawcza konfiguracji modeli z katalogu hf-configs.

Uruchomienie (w katalogu M1/jupyter):
	python data-visualization.py

Wynik: wypisany DataFrame oraz zapisane wykresy PNG w ./figures.
"""

import json
import math
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

CONFIG_DIR = Path(__file__).parent / "hf-configs"
OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

FIELDS = [
	"model_name",
	"model_type",
	"hidden_size",
	"intermediate_size",
	"num_hidden_layers",
	"num_attention_heads",
	"num_key_value_heads",
	"max_position_embeddings",
	"vocab_size",
	"estimated_params_millions",
]

def estimate_params(cfg: dict) -> float | None:
	"""Szacowanie liczby parametrów (bardzo uproszczone).

	Formula (dekoder layer approx):
		embedding = vocab_size * hidden_size
		per_layer = 4*H^2 (Q,K,V,O) + 2*H*I (MLP) => 4H^2 + 2HI
		total = embedding + L*(4H^2 + 2HI)
	Zwraca liczbę parametrów w milionach (float) lub None jeśli brak danych.
	"""
	try:
		H = cfg.get("hidden_size")
		I = cfg.get("intermediate_size")
		L = cfg.get("num_hidden_layers")
		V = cfg.get("vocab_size")
		if None in (H, I, L, V):
			return None
		total = V * H + L * (4 * H * H + 2 * H * I)
		return round(total / 1e6, 2)
	except Exception:
		return None

def load_configs(config_dir: Path) -> list[dict]:
	configs = []
	for path in sorted(config_dir.glob("*.json")):
		with path.open("r", encoding="utf-8") as f:
			data = json.load(f)
		row = {
			"model_name": path.stem.replace("-config", ""),
			"model_type": data.get("model_type"),
			"hidden_size": data.get("hidden_size"),
			"intermediate_size": data.get("intermediate_size"),
			"num_hidden_layers": data.get("num_hidden_layers"),
			"num_attention_heads": data.get("num_attention_heads"),
			"num_key_value_heads": data.get("num_key_value_heads"),
			"max_position_embeddings": data.get("max_position_embeddings"),
			"vocab_size": data.get("vocab_size"),
		}
		row["estimated_params_millions"] = estimate_params(row | data)
		configs.append(row)
	return configs

def build_dataframe(rows: list[dict]) -> pd.DataFrame:
	df = pd.DataFrame(rows, columns=FIELDS)
	return df.sort_values("estimated_params_millions", ascending=False, na_position="last")

def plot_bar(df: pd.DataFrame, column: str, title: str, ylabel: str):
	plt.figure(figsize=(8, 4))
	subset = df[["model_name", column]].dropna()
	plt.bar(subset["model_name"], subset[column], color="#4C72B0")
	plt.title(title)
	plt.ylabel(ylabel)
	plt.xticks(rotation=25, ha="right")
	plt.tight_layout()
	out = OUTPUT_DIR / f"{column}.png"
	plt.savefig(out)
	print(f"Zapisano wykres: {out}")
	plt.close()

def main():
	if not CONFIG_DIR.exists():
		raise SystemExit(f"Katalog z konfiguracjami nie istnieje: {CONFIG_DIR}")
	rows = load_configs(CONFIG_DIR)
	df = build_dataframe(rows)
	print("\nKonfiguracje modeli (wybrane pola):")
	print(df.to_string(index=False))

	# Podstawowe wykresy
	plot_bar(df, "hidden_size", "Hidden size", "Hidden size")
	plot_bar(df, "num_hidden_layers", "Liczba warstw", "Warstwy")
	plot_bar(df, "vocab_size", "Rozmiar słownika", "Vocab size")
	plot_bar(df, "estimated_params_millions", "Szacowana liczba parametrów (M)", "Parametry [M]")

	# Dodatkowe: stosunek heads / layers
	if {"num_attention_heads", "num_hidden_layers"}.issubset(df.columns):
		ratio_series = df.apply(
			lambda r: r["num_attention_heads"] / r["num_hidden_layers"] if (r["num_attention_heads"] and r["num_hidden_layers"]) else math.nan,
			axis=1,
		)
		df_ratio = pd.DataFrame({"model_name": df["model_name"], "heads_per_layer": ratio_series})
		plt.figure(figsize=(8, 4))
		plt.bar(df_ratio["model_name"], df_ratio["heads_per_layer"], color="#55A868")
		plt.title("Stosunek attention heads / warstwa")
		plt.ylabel("Heads per layer")
		plt.xticks(rotation=25, ha="right")
		plt.tight_layout()
		out = OUTPUT_DIR / "heads_per_layer.png"
		plt.savefig(out)
		print(f"Zapisano wykres: {out}")
		plt.close()

if __name__ == "__main__":
	main()


