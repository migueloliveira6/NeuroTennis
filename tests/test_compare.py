import pandas as pd
from datetime import datetime
from src import compare_predictions_results as comp


def test_normalize_name():
    assert comp.normalize_name('Rafaël Nadal') == 'rafael nadal'
    assert comp.normalize_name('  N.  Djokovic ') == 'n djokovic'


def make_results_df():
    data = [
        {'winner': 'Rafael Nadal', 'loser': 'Novak Djokovic', 'score': '6-3 6-4', 'tourney_name': 'Roland Garros'},
        {'winner': 'Roger Federer', 'loser': 'Andy Murray', 'score': '6-4 6-4', 'tourney_name': 'Wimbledon'}
    ]
    return pd.DataFrame(data)


def test_match_prediction_exact():
    pred = pd.Series({'Jogador 1': 'Rafael Nadal', 'Jogador 2': 'Novak Djokovic', 'Torneio': 'Roland Garros', 'Vencedor Previsto': 'Rafael Nadal'})
    res_df = make_results_df()
    matched = comp.match_prediction(pred, res_df)
    assert matched is not None
    assert matched['winner'] == 'Rafael Nadal'


def test_match_prediction_reversed_order():
    pred = pd.Series({'Jogador 1': 'Novak Djokovic', 'Jogador 2': 'Rafael Nadal', 'Torneio': 'Roland Garros', 'Vencedor Previsto': 'Rafael Nadal'})
    res_df = make_results_df()
    matched = comp.match_prediction(pred, res_df)
    assert matched is not None
    assert matched['winner'] == 'Rafael Nadal'


def test_match_prediction_no_match():
    pred = pd.Series({'Jogador 1': 'Some Player', 'Jogador 2': 'Other Player', 'Torneio': 'Unknown', 'Vencedor Previsto': 'Some Player'})
    res_df = make_results_df()
    matched = comp.match_prediction(pred, res_df)
    assert matched is None


def test_compare_adds_pred_date(tmp_path, monkeypatch):
    # Create a fake predictions CSV with a date in its filename
    df_pred = pd.DataFrame([
        {'Jogador 1': 'Rafael Nadal', 'Jogador 2': 'Novak Djokovic', 'Torneio': 'Roland Garros', 'Vencedor Previsto': 'Rafael Nadal', 'Confiança (%)': 80}
    ])
    pred_file = tmp_path / 'previsoes_tenis_2026-01-04.csv'
    df_pred.to_csv(pred_file, index=False)

    # Fake scraper that returns a matching result
    class FakeScraper:
        def scrape_results(self, results_date):
            return pd.DataFrame([
                {'winner': 'Rafael Nadal', 'loser': 'Novak Djokovic', 'score': '6-3 6-4', 'tourney_name': 'Roland Garros'}
            ])

    monkeypatch.setattr(comp, 'ResultsScraper', FakeScraper)

    df_out, summary = comp.compare(str(pred_file), datetime(2026, 1, 4))
    assert 'Pred_Date' in df_out.columns
    assert df_out['Pred_Date'].iloc[0] == '2026-01-04'
