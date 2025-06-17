import os
import pandas as pd
import glob
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# Leitura otimizada dos dados ATP
def readATPMatchesParseTime(dirname):
    allFiles = glob.glob(dirname + "/atp_matches_????.csv") + \
               glob.glob(dirname + "/atp_matches_qual_chall_????.csv")
    container = []
    for filen in allFiles:
        df = pd.read_csv(filen, encoding="ISO-8859-1", parse_dates=['tourney_date'], dayfirst=False)
        df['tourney_date'] = pd.to_datetime(df['tourney_date'], errors='coerce')
        container.append(df)
    matches = pd.concat(container, ignore_index=True)
    return matches

# Construção de head-to-head para todos os jogadores
def build_h2h_all(matches):
    h2h_data = {}
    for _, row in matches.iterrows():
        w = row['winner_name']
        l = row['loser_name']
        if w not in h2h_data:
            h2h_data[w] = {}
        if l not in h2h_data[w]:
            h2h_data[w][l] = {'w': 0, 'l': 0}
        h2h_data[w][l]['w'] += 1

        if l not in h2h_data:
            h2h_data[l] = {}
        if w not in h2h_data[l]:
            h2h_data[l][w] = {'w': 0, 'l': 0}
        h2h_data[l][w]['l'] += 1
    return h2h_data

# Extrair H2H vetorizado
def get_vectorized_h2h(df, h2h_data):
    wins, losses = [], []
    for p1, p2 in zip(df['player1'], df['player2']):
        data = h2h_data.get(p1, {}).get(p2, {'w': 0, 'l': 0})
        wins.append(data['w'])
        losses.append(data['l'])
    return wins, losses

# Codificação da superfície
def encode_surface(surface):
    surface_enc = [0, 0, 0]
    if surface == 'Clay': surface_enc[0] = 1
    elif surface == 'Hard': surface_enc[1] = 1
    elif surface == 'Grass': surface_enc[2] = 1
    return surface_enc

# Construção de estatísticas por superfície
def build_surface_stats(matches):
    stats = {}
    for _, row in matches.iterrows():
        for player_col, result_col in [('winner_name', 'w'), ('loser_name', 'l')]:
            player = row[player_col]
            surface = row['surface']
            if pd.isnull(player) or pd.isnull(surface):
                continue
            if player not in stats:
                stats[player] = {}
            if surface not in stats[player]:
                stats[player][surface] = {'w': 0, 'l': 0}
            stats[player][surface][result_col] += 1
    return stats

# função auxiliar para extrair taxa de vitória por superfície
def get_surface_features(row, surface_stats):
    p1 = row['player1']
    p2 = row['player2']
    surface = row['surface']

    def win_rate(player):
        data = surface_stats.get(player, {}).get(surface, {'w': 0, 'l': 0})
        w, l = data['w'], data['l']
        total = w + l
        return w / total if total > 0 else 0.5  # assume 50% se não houver dados

    return pd.Series({
        'p1_surface_wr': win_rate(p1),
        'p2_surface_wr': win_rate(p2)
    })

def add_elo_to_matches(matches, elo_data):
    """
    Adds Elo rating columns for both players to the matches DataFrame by merging with the provided elo_data DataFrame.

    Parameters:
        matches (pd.DataFrame): DataFrame containing match information with 'player1' and 'player2' columns.
        elo_data (pd.DataFrame): DataFrame containing player names and their Elo ratings for different surfaces and overall.

    Returns:
        pd.DataFrame: The input matches DataFrame with additional columns for player Elo ratings.
    """
    elo_cols = ['elo_Hard', 'elo_Clay', 'elo_Grass', 'elo_overall']
    # Merge Elo ratings for player1
    matches = matches.merge(
        elo_data[['name'] + elo_cols].rename(columns={
            'name': 'player1',
            'elo_Hard': 'player1_elo_hard',
            'elo_Clay': 'player1_elo_clay',
            'elo_Grass': 'player1_elo_grass',
            'elo_overall': 'player1_elo_overall'
        }),
        on='player1', how='left'
    )
    # Merge Elo ratings for player2
    matches = matches.merge(
        elo_data[['name'] + elo_cols].rename(columns={
            'name': 'player2',
            'elo_Hard': 'player2_elo_hard',
            'elo_Clay': 'player2_elo_clay',
            'elo_Grass': 'player2_elo_grass',
            'elo_overall': 'player2_elo_overall'
        }),
        on='player2', how='left'
    )
    # Fill missing Elo values with 1500
    for col in ['player1_elo_hard', 'player2_elo_hard', 'player1_elo_clay', 'player2_elo_clay',
                'player1_elo_grass', 'player2_elo_grass', 'player1_elo_overall', 'player2_elo_overall']:
        matches[col] = matches[col].fillna(1500)
    return matches
    return matches

# Função para prever o vencedor da partida
def predict_match(p1, p2, surface, model, scaler, surface_stats, elo_data):
    # Obter Elo ratings dos jogadores para as superfícies
    def get_elo(player, col):
        vals = elo_data.loc[elo_data['name'] == player, col].values
        return vals[0] if len(vals) > 0 else 1500  # Default Elo if not found
    player1_elo_clay = elo_data.loc[elo_data['name'] == p1, 'elo_Clay'].values[0] if not elo_data.loc[elo_data['name'] == p1, 'elo_Clay'].empty else 1500
    player2_elo_clay = elo_data.loc[elo_data['name'] == p2, 'elo_Clay'].values[0] if not elo_data.loc[elo_data['name'] == p2, 'elo_Clay'].empty else 1500
    player2_elo_hard = get_elo(p2, 'elo_Hard')

    player1_elo_clay = get_elo(p1, 'elo_Clay')
    player2_elo_clay = get_elo(p2, 'elo_Clay')
    player1_elo_overall = get_elo(p1, 'elo_overall') if not elo_data.loc[elo_data['name'] == p1, 'elo_overall'].empty else 1500
    player2_elo_overall = get_elo(p2, 'elo_overall') if not elo_data.loc[elo_data['name'] == p2, 'elo_overall'].empty else 1500
    player2_elo_grass = get_elo(p2, 'elo_Grass')

    player1_elo_overall = get_elo(p1, 'elo_overall')
    player2_elo_overall = get_elo(p2, 'elo_overall')

    # Codificar a superfície
    surface_enc = [1, 0, 0] if surface == 'Clay' else [0, 1, 0] if surface == 'Hard' else [0, 0, 1]

    # Obter a taxa de vitória por superfície
    def win_rate(player):
        data = surface_stats.get(player, {}).get(surface, {'w': 0, 'l': 0})
        w, l = data['w'], data['l']
        total = w + l
        return w / total if total > 0 else 0.5  # Assume 50% se não houver dados

    p1_surface_wr = win_rate(p1)
    p2_surface_wr = win_rate(p2)

    # Preparar o vetor de entrada para o modelo
    input_vector = [player1_elo_hard, player2_elo_hard, player1_elo_clay, player2_elo_clay, 
                    player1_elo_grass, player2_elo_grass, player1_elo_overall, player2_elo_overall] + surface_enc
    
    input_df = pd.DataFrame([input_vector], columns=[
        'player1_elo_hard', 'player2_elo_hard', 'player1_elo_clay', 'player2_elo_clay',
        'player1_elo_grass', 'player2_elo_grass', 'player1_elo_overall', 'player2_elo_overall',
        'surface_Clay', 'surface_Hard', 'surface_Grass'
    ])
    
    input_scaled = scaler.transform(input_df)

    prob = model.predict_proba(input_scaled)[0]
    winner = p1 if prob[1] > 0.5 else p2
    confidence = max(prob[1], prob[0])
    print(f"Previsão: {winner} com {confidence*100:.2f}% de confiança.")
    return winner, confidence

# Estatísticas por jogador
def show_player_stats(player_name, matches):
    surfaces = ['Clay', 'Hard', 'Grass']
    stats = []

    player_matches = matches[
        (matches['winner_name'] == player_name) | (matches['loser_name'] == player_name)
    ]

    print(f"\n📈 Estatísticas de {player_name} por superfície:")
    for surface in surfaces:
        surface_matches = player_matches[player_matches['surface'] == surface]
        wins = surface_matches[surface_matches['winner_name'] == player_name].shape[0]
        losses = surface_matches[surface_matches['loser_name'] == player_name].shape[0]
        total = wins + losses
        win_rate = (wins / total * 100) if total > 0 else 0
        stats.append({
            'Superfície': surface,
            'Jogos': total,
            'Vitórias': wins,
            'Derrotas': losses,
            'Taxa de Vitória (%)': round(win_rate, 2)
        })

    df_stats = pd.DataFrame(stats)
    print(df_stats.to_string(index=False))

    total_wins = player_matches[player_matches['winner_name'] == player_name].shape[0]
    total_losses = player_matches[player_matches['loser_name'] == player_name].shape[0]
    total_games = total_wins + total_losses
    win_rate_total = (total_wins / total_games * 100) if total_games > 0 else 0
    print(f"\n🏁 Total de jogos: {total_games} | Vitórias: {total_wins} | Derrotas: {total_losses} | Taxa de vitória: {win_rate_total:.2f}%")

    print("\n🕒 Últimos 5 jogos:")
    recent = player_matches.sort_values('tourney_date', ascending=False).head(5)
    for _, row in recent.iterrows():
        print(f"Data: {row['tourney_date'].date()}, Oponente: {row['loser_name'] if row['winner_name'] == player_name else row['winner_name']}, Duração: {row['minutes']} minutos, Aces: {row['w_ace'] if row['winner_name'] == player_name else row['l_ace']}")

    if 'round' in matches.columns:
        finals = matches[(matches['round'] == 'F') & (matches['winner_name'] == player_name)]
        print(f"\n🏆 Títulos ganhos (finais vencidas): {len(finals)}")
        finals_per_year = finals['tourney_date'].dt.year.value_counts().sort_index()
        for year, count in finals_per_year.items():
            print(f"  {year}: {count} título(s)")

# Menu principal
def main_menu():
    dirname = 'D:/projetos/Tenis ML-AI/data/tennis_atp'
    rank_file = dirname + '/atp_rankings_current.csv'
    elo_data = pd.read_csv('D:/projetos/Tenis ML-AI/notebooks/players_elo_by_surface_filtered.csv')

    if os.path.exists('D:/projetos/Tenis ML-AI/models/modelo_treinado_elo.pkl') and os.path.exists('D:/projetos/Tenis ML-AI/models/dados_preparados_elo.pkl') and os.path.exists('D:/projetos/Tenis ML-AI/models/scaler_treinado_elo.pkl'):
        print("📂 A carregar modelo e dados do disco...")
        model = joblib.load('D:/projetos/Tenis ML-AI/models/modelo_treinado_elo.pkl')
        scaler = joblib.load('D:/projetos/Tenis ML-AI/models/scaler_treinado_elo.pkl')
        matches, df_encoded, rank_df, h2h_data, surface_stats = joblib.load('D:/projetos/Tenis ML-AI/models/dados_preparados_elo.pkl')
    else:
        print("🔄 A preparar dados e treinar modelo (primeira vez)...")
        matches = readATPMatchesParseTime(dirname)
        rank_df = pd.read_csv(rank_file)
        rank_df = rank_df.rename(columns={"player": "player_name"})

        base_df = matches[['surface', 'winner_name', 'loser_name', 'winner_rank', 'loser_rank']].dropna()

        # Build surface_stats before using it
        surface_stats = build_surface_stats(matches)

        print("📊 A gerar histórico H2H...")
        # Build h2h_data before using it
        h2h_data = build_h2h_all(matches)

        df1 = base_df.copy()
        df1['player1'] = df1['winner_name']
        df1['player2'] = df1['loser_name']
        df1['player1_rank'] = df1['winner_rank']
        df1['player2_rank'] = df1['loser_rank']
        df1['target'] = 1

        df2 = base_df.copy()
        df2['player1'] = df2['loser_name']
        df2['player2'] = df2['winner_name']
        df2['player1_rank'] = df2['loser_rank']
        df2['player2_rank'] = df2['winner_rank']
        df2['target'] = 0

        df_combined = pd.concat([df1, df2], ignore_index=True)
        df_combined = df_combined[['surface', 'player1', 'player2', 'player1_rank', 'player2_rank', 'target']]

        surface_features = df_combined.apply(lambda row: get_surface_features(row, surface_stats), axis=1)
        df_combined = pd.concat([df_combined, surface_features], axis=1)
        # Adiciona colunas de Elo antes de aplicar get_dummies
        df_combined = add_elo_to_matches(df_combined, elo_data)
        h2h_wins, h2h_losses = get_vectorized_h2h(df_combined, h2h_data)
        df_combined['h2h_wins'] = h2h_wins
        df_combined['h2h_losses'] = h2h_losses

        surface_features = df_combined.apply(lambda row: get_surface_features(row, surface_stats), axis=1)
        df_combined = pd.concat([df_combined, surface_features], axis=1)

        # Adiciona colunas de Elo ao df_combined
        df_combined = add_elo_to_matches(df_combined, elo_data)

        df_encoded = pd.get_dummies(df_combined, columns=['surface'])
        feature_cols = ['player1_rank', 'player2_rank', 'h2h_wins', 'h2h_losses',
                        'p1_surface_wr', 'p2_surface_wr', 'player1_elo_hard', 'player2_elo_hard', 
                        'player1_elo_clay', 'player2_elo_clay', 'player1_elo_grass', 'player2_elo_grass',
                        'player1_elo_overall', 'player2_elo_overall', 
                        'surface_Clay', 'surface_Hard', 'surface_Grass']
        print("🔄 A preparar dados para treino...")
        df_encoded = df_encoded[df_encoded[feature_cols].notnull().all(axis=1)]
        X = df_encoded[feature_cols]
        y = df_encoded['target']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=120, max_depth=20, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        print("✅ Modelo treinado com acuraccy:", accuracy_score(y_test, model.predict(X_test)))

        print("🔄 A avaliar modelo com acurácia...")
        f1 = f1_score(y_test, model.predict(X_test))
        print(f"F1-score: {f1:.2f}")

        print("🔄 A guardar modelo e dados preparados...")

        joblib.dump(model, 'D:/projetos/Tenis ML-AI/models/modelo_treinado_elo.pkl')
        joblib.dump(scaler, 'D:/projetos/Tenis ML-AI/models/scaler_treinado_elo.pkl')
        joblib.dump((matches, df_encoded, rank_df, h2h_data, surface_stats), 'D:/projetos/Tenis ML-AI/models/dados_preparados_elo.pkl')

        print("✅ Modelo e dados preparados salvos com sucesso!")

    while True:
        print("\n========= MENU =========")
        print("1. Fazer previsão de partida")
        print("2. Ver exemplos de previsão")
        print("3. Ver estatísticas de jogador")
        print("4. Sair")
        choice = input("Escolha: ")

        if choice == '1':
            p1 = input("Nome do jogador 1: ").strip()
            p2 = input("Nome do jogador 2: ").strip()
            surface = input("Superfície (Clay, Hard, Grass): ").strip().capitalize()
            winner = predict_match(p1, p2, surface, model, scaler, surface_stats, elo_data)
            print(f'O vencedor previsto é: {winner}')

        elif choice == '2':
            print("A gerar exemplos de previsão...")
            sample = df_encoded.sample(8, random_state=1)
            for i, row in sample.iterrows():
                p1 = row['player1']
                p2 = row['player2']
                if row.get('surface_Clay', 0): surface = 'Clay'
                elif row.get('surface_Hard', 0): surface = 'Hard'
                elif row.get('surface_Grass', 0): surface = 'Grass'
                else: surface = 'Hard'
                print(f"\nExemplo {i + 1}: {p1} vs {p2} em {surface}")
                winner = predict_match(p1, p2, surface, model, scaler, rank_df, h2h_data, surface_stats, elo_data)
                print(f'O vencedor previsto é: {winner}')

        elif choice == '3':
            p = input("Nome do jogador: ").strip()
            show_player_stats(p, matches)

        elif choice == '4':
            print("👋 Saindo...")
            break
        else:
            print("❌ Opção inválida.")

# Executar
if __name__ == '__main__':
    main_menu()
