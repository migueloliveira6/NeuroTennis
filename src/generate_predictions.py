# scripts/generate_predictions.py
import datetime
import os
import json
import pandas as pd
from glob import glob
import re

# === CONFIGURAÇÕES ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "previsoes")
DOCS_DATA_DIR = os.path.join(BASE_DIR, "docs", "predicts")

# Cria pastas se não existirem
os.makedirs(DOCS_DATA_DIR, exist_ok=True)

# === 1️⃣ Localizar o CSV mais recente de previsões ===
pattern = os.path.join(NOTEBOOKS_DIR, "previsoes_tenis_*.csv")
files = glob(pattern)
if not files:
    raise FileNotFoundError("Nenhum arquivo de previsões encontrado em previsoes/")

date_pattern = re.compile(r"^previsoes_tenis_(\d{4}-\d{2}-\d{2})\.csv$")
dated_files = []
for path in files:
    base = os.path.basename(path)
    match = date_pattern.match(base)
    if not match:
        continue
    date_str = match.group(1)
    dated_files.append((datetime.datetime.strptime(date_str, "%Y-%m-%d"), path))

if not dated_files:
    raise FileNotFoundError("Nenhum arquivo de previsões com data válido encontrado em previsoes/")

csv_path = max(dated_files, key=lambda item: item[0])[1]

print(f"📄 Carregando previsões do arquivo: {csv_path}")
df = pd.read_csv(csv_path)

# === 2️⃣ Garantir colunas esperadas ===
expected_cols = [
    "Torneio", "Jogador 1", "Jogador 2", "Vencedor Previsto",
    "Confiança (%)", "ELO Diff", "H2H", "Odd 1", "Odd 2",
    "Superfície", "Valor Aposta", "ROI Esperado (%)"
]
missing_cols = [c for c in expected_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Colunas ausentes no CSV: {missing_cols}")

# === 4️⃣ Salvar como JSON para o dashboard ===
output_json_path = os.path.join(DOCS_DATA_DIR, "predictions.json")
df.to_json(output_json_path, orient="records", force_ascii=False, indent=4)

print(f"✅ Previsões exportadas para {output_json_path}")
print(f"🔗 Pronto para deploy no GitHub Pages!")
