import os
import pandas as pd
import glob
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from operator import itemgetter

# Função para fazer o parse das datas
def parse(t):
    ret = []
    for ts in t:
        try:
            string = str(ts)
            tsdt = datetime.date(int(string[:4]), int(string[4:6]), int(string[6:]))
        except:
            tsdt = datetime.date(1900, 1, 1)
        ret.append(tsdt)
    return ret

# Leitura dos dados de partidas ATP
def readATPMatchesParseTime(dirname):
    allFiles = glob.glob(dirname + "/atp_matches_" + "????.csv")
    container = []
    for filen in allFiles:
        df = pd.read_csv(filen,
                         index_col=None,
                         header=0,
                         encoding="ISO-8859-1",
                         parse_dates=[5],
                         date_parser=lambda t: parse(t))
        container.append(df)
    matches = pd.concat(container, ignore_index=True)
    return matches

# Histórico head-to-head
def geth2hforplayer(matches, name):
    matches = matches[(matches['winner_name'] == name) | (matches['loser_name'] == name)]
    h2hs = {}
    for _, match in matches.iterrows():
        if match['winner_name'] == name:
            opponent = match['loser_name']
            if opponent not in h2hs:
                h2hs[opponent] = {'w': 1, 'l': 0}
            else:
                h2hs[opponent]['w'] += 1
        elif match['loser_name'] == name:
            opponent = match['winner_name']
            if opponent not in h2hs:
                h2hs[opponent] = {'w': 0, 'l': 1}
            else:
                h2hs[opponent]['l'] += 1
    return h2hs

# Criar features de H2H para um par de jogadores
def extract_h2h_feature(h2h_data, p1, p2):
    h2h_p1 = h2h_data.get(p1, {})
    if p2 in h2h_p1:
        wins = h2h_p1[p2].get('w', 0)
        losses = h2h_p1[p2].get('l', 0)
    else:
        wins = losses = 0
    return wins, losses

# Codificação da superfície
def encode_surface(surface):
    surface_enc = [0, 0, 0]  # Clay, Hard, Grass
    if surface == 'Clay':
        surface_enc[0] = 1
    elif surface == 'Hard':
        surface_enc[1] = 1
    elif surface == 'Grass':
        surface_enc[2] = 1
    return surface_enc

# Função de previsão
def predict_match(p1, p2, surface, model, scaler, rank_df, h2h_data):
    try:
        r1 = rank_df.loc[rank_df['player_name'] == p1, 'rank'].values[0]
    except:
        r1 = 2000
    try:
        r2 = rank_df.loc[rank_df['player_name'] == p2, 'rank'].values[0]
    except:
        r2 = 2000

    wins, losses = extract_h2h_feature(h2h_data, p1, p2)

    surface_enc = encode_surface(surface)
    input_vector = [r1, r2, wins, losses] + surface_enc
    input_scaled = scaler.transform([input_vector])

    prob = model.predict_proba(input_scaled)[0]
    winner = p1 if prob[1] > 0.5 else p2
    confidence = max(prob[1], prob[0])

    print(f"Previsão: {winner} com {confidence*100:.2f}% de confiança.")

def show_player_stats(player_name, matches):
    surfaces = ['Clay', 'Hard', 'Grass']
    stats = []

    for surface in surfaces:
        surface_matches = matches[matches['surface'] == surface]

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
    print(f"\n📈 Estatísticas de {player_name} por superfície:")
    print(df_stats.to_string(index=False))
    
# Menu principal
def main_menu():

    dirname = 'D:/projetos/Tenis ML-AI/data/tennis_atp'
    rank_file = dirname + '/atp_rankings_current.csv'

    print("🔄 A carregar dados...")
    matches = readATPMatchesParseTime(dirname)
    rank_df = pd.read_csv(rank_file)
    rank_df = rank_df.rename(columns={"player": "player_name", "rank": "rank"})

    # Filtrar e preparar dados
    base_df = matches[['surface', 'winner_name', 'loser_name', 'winner_rank', 'loser_rank']].dropna()

    # Criar duas versões dos dados (p1 vence e p1 perde)
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

    # H2H
    print("A gerar histórico H2H...")
    players = pd.concat([matches['winner_name'], matches['loser_name']]).dropna().unique()
    h2h_data = {name: geth2hforplayer(matches, name) for name in players}

    df_combined['h2h_wins'] = df_combined.apply(
        lambda row: extract_h2h_feature(h2h_data, row['player1'], row['player2'])[0], axis=1
    )
    df_combined['h2h_losses'] = df_combined.apply(
        lambda row: extract_h2h_feature(h2h_data, row['player1'], row['player2'])[1], axis=1
    )

    # Codificar superfície
    df_encoded = pd.get_dummies(df_combined, columns=['surface'])
    feature_cols = ['player1_rank', 'player2_rank', 'h2h_wins', 'h2h_losses',
                    'surface_Clay', 'surface_Hard', 'surface_Grass']
    df_encoded = df_encoded[df_encoded[feature_cols].notnull().all(axis=1)]

    X = df_encoded[feature_cols]
    y = df_encoded['target']

    # Normalizar e treinar modelo
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    print("Modelo treinado com acuccary:", accuracy_score(y_test, model.predict(X_test)))

    # Menu de previsão
    while True:
        print("1. Fazer previsão de partida")
        print("2. Ver exemplos de previsão")
        print("3. Ver estatísticas de jogador")
        print("4. Sair")
        choice = input("Escolha: ")

        if choice == '1':
            p1 = input("Nome do jogador 1: ").strip()
            p2 = input("Nome do jogador 2: ").strip()
            surface = input("Superfície (Clay, Hard, Grass): ").strip().capitalize()
            predict_match(p1, p2, surface, model, scaler, rank_df, h2h_data)

        elif choice == '2':
            print("A gerar exemplos de previsão...")
            example_matches = df_encoded.sample(5, random_state=1)  # 5 exemplos aleatórios
            for i, row in example_matches.iterrows():
                p1 = row['player1']
                p2 = row['player2']
                if 'surface_Clay' in row and row['surface_Clay'] == 1:
                    surface = 'Clay'
                elif 'surface_Hard' in row and row['surface_Hard'] == 1:
                    surface = 'Hard'
                elif 'surface_Grass' in row and row['surface_Grass'] == 1:
                    surface = 'Grass'
                else:
                    surface = 'Hard' # Default to Hard if not specified

                print(f"\nExemplo {i + 1}: {p1} vs {p2} em {surface}")
                predict_match(p1, p2, surface, model, scaler, rank_df, h2h_data)
        elif choice == '3':
            p = input("Nome do jogador: ").strip()
            show_player_stats(p, matches)

        elif choice == '4':
            print("Saindo...")
            break
        else:
            print("❌ Opção inválida.")

# Executar
if __name__ == '__main__':
    main_menu()
# This code is a complete implementation of a tennis match prediction system using machine learning.
# It includes data loading, preprocessing, feature extraction, model training, and prediction functionalities.