# scripts/generate_predictions.py
import os
import json
import pandas as pd
from datetime import datetime
from model_elo_xgboost import TennisPredictor

# === CONFIGURAÇÕES ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks", "previsoes")
DOCS_DATA_DIR = os.path.join(BASE_DIR, "docs", "predicts")

# Cria pastas se não existirem
os.makedirs(DOCS_DATA_DIR, exist_ok=True)

# === 1️⃣ Localizar o CSV mais recente de previsões ===
csv_files = [f for f in os.listdir(NOTEBOOKS_DIR) if f.startswith("previsoes_tenis_v2_") and f.endswith(".csv")]
if not csv_files:
    raise FileNotFoundError("Nenhum arquivo de previsões encontrado em notebooks/previsoes/")

# Ordena por data (caso existam vários)
csv_files.sort(reverse=True)
latest_csv = csv_files[0]
csv_path = os.path.join(NOTEBOOKS_DIR, latest_csv)

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

# === 3️⃣ (Opcional) Validar ou recalcular previsões com o modelo ===
# Se quiser revalidar o modelo, basta descomentar as linhas abaixo.
# predictor = TennisPredictor()
# predictor.load_saved_model()
# # Exemplo: recalcular confiança com base no modelo salvo
# for i, row in df.iterrows():
#     winner, conf, _ = predictor.predict_match(row["Jogador 1"], row["Jogador 2"], row["Superfície"])
#     if winner:
#         df.at[i, "Vencedor Previsto"] = winner
#         df.at[i, "Confiança (%)"] = round(conf * 100, 1)

# === 4️⃣ Salvar como JSON para o dashboard ===
output_json_path = os.path.join(DOCS_DATA_DIR, "predictions.json")
df.to_json(output_json_path, orient="records", force_ascii=False, indent=4)

print(f"✅ Previsões exportadas para {output_json_path}")
print(f"🔗 Pronto para deploy no GitHub Pages!")
