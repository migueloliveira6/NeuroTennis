#!/usr/bin/env python3
"""
Comparador de Previsões vs Resultados
- Localiza o arquivo de previsões mais recente em `PREVISOES_PATH` (ou usa --pred-file)
- Determina a data alvo (pode ser extraída do nome do ficheiro ou passada via --results-date)
- Faz scraping dos resultados com `scraper_results.ResultsScraper.scrape_results`
- Normaliza nomes e faz o matching das partidas
- Calcula métricas (matched, accuracy) e salva CSV de comparação
"""

import os
import re
import argparse
from datetime import datetime, timedelta
import unicodedata
import pandas as pd
import requests
from typing import Optional

# Importar função do scraper local
try:
    from scraper_results import ResultsScraper
except Exception:
    # Tentar ajustar import se for executado de outro working dir
    import sys
    sys.path.append(os.path.dirname(__file__))
    from scraper_results import ResultsScraper


def find_latest_prediction_file(previsoes_path: str) -> Optional[str]:
    files = [f for f in os.listdir(previsoes_path) if f.startswith('previsoes_tenis') and f.endswith('.csv')]
    if not files:
        return None
    files_full = [os.path.join(previsoes_path, f) for f in files]
    files_full.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files_full[0]


def find_prediction_file_for_date(previsoes_path: str, target_date: datetime) -> Optional[str]:
    files = [f for f in os.listdir(previsoes_path) if f.startswith('previsoes_tenis') and f.endswith('.csv')]
    if not files:
        return None

    target = target_date.date()
    candidates = []
    for f in files:
        dt = extract_date_from_filename(f)
        if dt and dt.date() == target:
            candidates.append(os.path.join(previsoes_path, f))

    if not candidates:
        return None

    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def extract_date_from_filename(filename: str) -> Optional[datetime]:
    m = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    if m:
        try:
            return datetime.strptime(m.group(1), '%Y-%m-%d')
        except ValueError:
            return None
    return None


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ''
    s = name.strip().lower()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    # Keep letters, numbers and spaces
    s = re.sub(r'[^a-z0-9 ]+', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def match_prediction(pred_row: pd.Series, results_df: pd.DataFrame) -> Optional[pd.Series]:
    p1 = normalize_name(pred_row.get('Jogador 1') or pred_row.get('player1') or '')
    p2 = normalize_name(pred_row.get('Jogador 2') or pred_row.get('player2') or '')
    torneio_pred = normalize_name(pred_row.get('Torneio') or pred_row.get('tourney') or '')

    if not p1 or not p2:
        return None

    for _, res in results_df.iterrows():
        winner = normalize_name(res.get('winner', ''))
        loser = normalize_name(res.get('loser', ''))
        tourney = normalize_name(res.get('tourney_name', ''))

        # verificar se os jogadores coincidem (independente da ordem)
        if {p1, p2} == {winner, loser}:
            # preferir que o torneio coincida ou que seja substring
            if torneio_pred and torneio_pred in tourney:
                return res
            else:
                # se não houver match de torneio, ainda podemos aceitar
                return res

    # Nenhum match exato encontrado
    return None


def compare(pred_file: Optional[str], results_date: datetime, out_dir: str = 'previsoes', df_pred: Optional[pd.DataFrame] = None, pred_file_label: Optional[str] = None):
    if df_pred is None:
        if not pred_file:
            raise ValueError("pred_file é obrigatório quando df_pred não é fornecido")
        df_pred = pd.read_csv(pred_file)
        pred_file_label = pred_file_label or os.path.basename(pred_file)
    else:
        pred_file_label = pred_file_label or (os.path.basename(pred_file) if pred_file else "predictions.json")

    # Scrape results for the results_date
    scraper = ResultsScraper()
    results_df = scraper.scrape_results(results_date)

    if results_df.empty:
        print('⚠️  Nenhum resultado encontrado para a data:', results_date.strftime('%Y-%m-%d'))
        return

    # Prepare results DataFrame
    out_rows = []
    total = 0
    correct = 0
    matched = 0

    for _, pred in df_pred.iterrows():
        total += 1
        matched_row = match_prediction(pred, results_df)

        if matched_row is not None:
            matched += 1
            actual_winner = matched_row.get('winner')
            actual_loser = matched_row.get('loser')
            score = matched_row.get('score')
            tourney = matched_row.get('tourney_name')
            is_correct = normalize_name(pred.get('Vencedor Previsto', '')) == normalize_name(actual_winner)
            if is_correct:
                correct += 1
        else:
            actual_winner = None
            actual_loser = None
            score = None
            tourney = None
            is_correct = False

        out_rows.append({
            'Pred_File': pred_file_label,
            'Pred_Date': results_date.strftime('%Y-%m-%d'),
            'Pred_Torneio': pred.get('Torneio'),
            'Jogador 1': pred.get('Jogador 1'),
            'Jogador 2': pred.get('Jogador 2'),
            'Vencedor Previsto': pred.get('Vencedor Previsto'),
            'Confiança (%)': pred.get('Confiança (%)'),
            'Actual_Winner': actual_winner,
            'Actual_Loser': actual_loser,
            'Score': score,
            'Match_Tourney': tourney,
            'Matched': bool(matched_row is not None),
            'Correct': bool(is_correct)
        })

    df_out = pd.DataFrame(out_rows)

    summary = {
        'predictions_total': total,
        'matched': matched,
        'correct': correct,
        'accuracy': (correct / matched) if matched else None,
        'coverage': (matched / total) if total else None
    }

    # Salvar CSV
    date_str = results_date.strftime('%Y-%m-%d')
    out_path = os.path.join(out_dir, f'comparacao_previsoes_{date_str}.csv')
    os.makedirs(out_dir, exist_ok=True)
    df_out.to_csv(out_path, index=False)

    print('\n✅ Comparação concluída')
    print(f"Total matches: {total}")
    print(f"Previsões encontradas: {matched}")
    print(f"Previsões corretas: {correct}")
    if summary['accuracy'] is not None:
        print(f"Accuracy (sobre matched): {summary['accuracy']:.2%}")
    if summary['coverage'] is not None:
        print(f"Cobertura (matched/total): {summary['coverage']:.2%}")

    print(f"CSV salvo em: {out_path}\n")

    return df_out, summary


def _load_predictions_from_json_url(pred_json_url: str) -> pd.DataFrame:
    res = requests.get(pred_json_url, timeout=30)
    res.raise_for_status()
    data = res.json()
    if not isinstance(data, list):
        raise ValueError("JSON de previsões inválido: esperado uma lista")
    return pd.DataFrame(data)


def _load_predictions_from_json_file(pred_json_file: str) -> pd.DataFrame:
    df = pd.read_json(pred_json_file)
    if df.empty:
        raise ValueError("JSON de previsões vazio")
    return df


def main():
    parser = argparse.ArgumentParser(description='Comparar previsões com resultados reais')
    parser.add_argument('--pred-file', type=str, help='Arquivo de previsões CSV (se não fornecido, procura o ficheiro do dia alvo em PREVISOES_PATH)')
    parser.add_argument('--preds-path', type=str, default=os.getenv('PREVISOES_PATH', 'previsoes'), help='Pasta onde estão as previsões')
    parser.add_argument('--results-date', type=str, help='Data dos resultados (YYYY-MM-DD). Se omitido, usa ontem')
    parser.add_argument('--out-dir', type=str, default='previsoes', help='Diretório para salvar o CSV de comparação')
    parser.add_argument('--pred-json-url', type=str, help='URL do predictions.json (ex: gh-pages)')
    parser.add_argument('--pred-json-file', type=str, help='Caminho local para predictions.json')

    args = parser.parse_args()

    pred_json_url = args.pred_json_url or os.getenv('PREDICTIONS_JSON_URL')
    pred_json_file = args.pred_json_file

    pred_file = args.pred_file
    df_pred = None
    pred_file_label = None

    # Default operacional: comparar sempre o dia anterior
    yesterday = datetime.now() - timedelta(days=1)
    default_results_date = datetime.combine(yesterday.date(), datetime.min.time())

    # Data dos resultados
    if args.results_date:
        results_date = datetime.strptime(args.results_date, '%Y-%m-%d')
    else:
        results_date = default_results_date

    if pred_json_file:
        df_pred = _load_predictions_from_json_file(pred_json_file)
        pred_file_label = os.path.basename(pred_json_file)
    elif pred_json_url:
        df_pred = _load_predictions_from_json_url(pred_json_url)
        pred_file_label = pred_json_url
    else:
        if not pred_file:
            pred_file = find_prediction_file_for_date(args.preds_path, results_date)
            if not pred_file:
                raise FileNotFoundError(
                    f"Nenhum arquivo de previsões encontrado para {results_date.strftime('%Y-%m-%d')} em {args.preds_path}"
                )
        elif not args.results_date:
            # Se o ficheiro for fornecido manualmente e a data não for explícita,
            # usar a data do nome do ficheiro quando disponível.
            dt = extract_date_from_filename(os.path.basename(pred_file))
            if dt:
                results_date = dt

    result = compare(pred_file, results_date, args.out_dir, df_pred=df_pred, pred_file_label=pred_file_label)
    if not result:
        return

    df_out, summary = result

if __name__ == '__main__':
    main()
