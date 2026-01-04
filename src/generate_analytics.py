# scripts/generate_analysis.py

import os
import json
import pandas as pd
from datetime import datetime
from collections import defaultdict

# === CONFIGURAÇÕES ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPARISONS_DIR = os.path.join(BASE_DIR, "previsoes")
DOCS_DATA_DIR = os.path.join(BASE_DIR, "docs", "analytics")

# Cria pastas se não existirem
os.makedirs(DOCS_DATA_DIR, exist_ok=True)

# === 1️⃣ Localizar CSVs de comparação ===
csv_files = [f for f in os.listdir(COMPARISONS_DIR) if f.startswith("comparacao_previsoes") and f.endswith(".csv")]

if not csv_files:
    print("⚠️  Nenhum arquivo de comparação encontrado. Execute compare_predictions_results.py primeiro.")
    exit(0)

print(f"📊 Encontrados {len(csv_files)} arquivo(s) de comparação")

# === 2️⃣ Carregar e consolidar todos os CSVs ===
all_data = []
for csv_file in csv_files:
    csv_path = os.path.join(COMPARISONS_DIR, csv_file)
    df_temp = pd.read_csv(csv_path)
    all_data.append(df_temp)
    print(f"   ✓ {csv_file}")

df = pd.concat(all_data, ignore_index=True)

# === 3️⃣ Validar colunas esperadas ===
expected_cols = [
    "Pred_File", "Pred_Date", "Pred_Torneio", "Jogador 1", "Jogador 2",
    "Vencedor Previsto", "Confiança (%)", "Actual_Winner",
    "Matched", "Correct"
]

missing_cols = [c for c in expected_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Colunas ausentes no CSV: {missing_cols}")

# === 4️⃣ Filtrar apenas previsões matched (que tiveram resultado) ===
df_matched = df[df['Matched'] == True].copy()

if df_matched.empty:
    print("⚠️  Nenhuma previsão matched encontrada. Não há dados para análise.")
    exit(0)

print(f"\n📈 Total de previsões matched: {len(df_matched)}")

# === 5️⃣ Calcular estatísticas globais ===
total_predictions = len(df_matched)
total_correct = df_matched['Correct'].sum()
total_wrong = total_predictions - total_correct
accuracy_global = (total_correct / total_predictions * 100) if total_predictions > 0 else 0

global_stats = {
    "total_predictions": int(total_predictions),
    "correct": int(total_correct),
    "wrong": int(total_wrong),
    "accuracy": round(accuracy_global, 2),
    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

print(f"   ✓ Taxa de acerto global: {accuracy_global:.1f}%")

# === 6️⃣ Extrair data do nome do arquivo ===
def extract_date_from_filename(filename):
    """Extrai data do formato comparacao_previsoes_YYYY-MM-DD.csv"""
    import re
    match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    if match:
        return match.group(1)
    return None

# Adicionar coluna de data ao DataFrame
if 'Pred_Date' in df_matched.columns:
    # Use explicit Pred_Date coluna se disponível
    df_matched['date'] = df_matched['Pred_Date'].astype(str)
else:
    df_matched['date'] = df_matched['Pred_File'].apply(extract_date_from_filename)

# === 7️⃣ Agregação por data ===
accuracy_by_date = []
for date in df_matched['date'].dropna().unique():
    df_date = df_matched[df_matched['date'] == date]
    total = len(df_date)
    correct = df_date['Correct'].sum()
    wrong = total - correct
    acc = (correct / total * 100) if total > 0 else 0
    
    accuracy_by_date.append({
        "date": date,
        "total_predictions": int(total),
        "correct": int(correct),
        "wrong": int(wrong),
        "accuracy": round(acc, 2)
    })

# Ordenar por data
accuracy_by_date.sort(key=lambda x: x['date'])

print(f"   ✓ {len(accuracy_by_date)} dias com dados")

# === 8️⃣ Agregação por superfície ===
accuracy_by_surface = []
for surface in df_matched['Pred_Torneio'].dropna().unique():
    df_surface = df_matched[df_matched['Pred_Torneio'] == surface]
    total = len(df_surface)
    correct = df_surface['Correct'].sum()
    acc = (correct / total * 100) if total > 0 else 0
    
    if total >= 3:  # Apenas superfícies com pelo menos 3 previsões
        accuracy_by_surface.append({
            "surface": surface,
            "total_predictions": int(total),
            "correct": int(correct),
            "accuracy": round(acc, 2)
        })

accuracy_by_surface.sort(key=lambda x: x['accuracy'], reverse=True)

print(f"   ✓ {len(accuracy_by_surface)} torneios/superfícies analisados")

# === 9️⃣ Agregação por faixa de confiança ===
def get_confidence_bucket(conf):
    if conf >= 80:
        return "80-100%"
    elif conf >= 70:
        return "70-79%"
    elif conf >= 60:
        return "60-69%"
    elif conf >= 50:
        return "50-59%"
    else:
        return "<50%"

df_matched['confidence_bucket'] = df_matched['Confiança (%)'].apply(get_confidence_bucket)

accuracy_by_confidence = []
for bucket in ["80-100%", "70-79%", "60-69%", "50-59%", "<50%"]:
    df_bucket = df_matched[df_matched['confidence_bucket'] == bucket]
    total = len(df_bucket)
    if total > 0:
        correct = df_bucket['Correct'].sum()
        acc = (correct / total * 100) if total > 0 else 0
        
        accuracy_by_confidence.append({
            "confidence_range": bucket,
            "total_predictions": int(total),
            "correct": int(correct),
            "accuracy": round(acc, 2)
        })

print(f"   ✓ Análise por faixa de confiança concluída")

# === 🔟 Salvar JSONs ===
output_files = {
    "global_stats.json": global_stats,
    "accuracy_by_date.json": accuracy_by_date,
    "accuracy_by_surface.json": accuracy_by_surface,
    "accuracy_by_confidence.json": accuracy_by_confidence
}

for filename, data in output_files.items():
    output_path = os.path.join(DOCS_DATA_DIR, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"   ✓ {filename}")

print(f"\n✅ Análise concluída! Arquivos salvos em {DOCS_DATA_DIR}")
print(f"🔗 Pronto para deploy no GitHub Pages!")
