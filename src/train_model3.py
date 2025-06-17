import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from datetime import datetime

# Configurações globais

MODEL_PATH = 'D:/projetos/Tenis ML-AI/models/'
INITIAL_ELO = 1500

class TennisPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.matches = None
        self.player_stats = None
        self.surface_stats = None
        self.h2h_data = None
        
    def load_data(self):
        """Carrega os dados do arquivo combinado com ELO por superfície"""
        print("Carregando dados de partidas...")
        self.matches = pd.read_csv('atp_matches_with_surface_elo.csv', 
                                 parse_dates=['tourney_date'],
                                 low_memory=False)
        
        # Garantir que temos as colunas de ELO por superfície
        required_columns = ['winner_surface_elo', 'loser_surface_elo']
        if not all(col in self.matches.columns for col in required_columns):
            raise ValueError("Dataset não contém colunas de ELO por superfície")
            
        print(f"Dados carregados: {len(self.matches)} partidas")
        
    def preprocess_data(self):
        """Prepara os dados para treinamento com ambas as classes"""
        print("\nPreprocessando dados...")
        
        # Filtrar partidas válidas
        valid_matches = self.matches[
            (self.matches['winner_name'].notna()) & 
            (self.matches['loser_name'].notna()) &
            (self.matches['surface'].notna())
        ].copy()
        
        # Construir estatísticas de superfície
        self._build_surface_stats(valid_matches)
        
        # Construir histórico H2H
        self._build_h2h_data(valid_matches)
        
        # Criar dataset com ambas as perspectivas (vencedor e perdedor)
        print("Criando dataset balanceado...")
        features_winner = []
        features_loser = []
        
        for _, row in valid_matches.iterrows():
            # Perspectiva do vencedor (target=1)
            feature_win = self._extract_features(row, perspective='winner')
            features_winner.append(feature_win)
            
            # Perspectiva do perdedor (target=0)
            feature_lose = self._extract_features(row, perspective='loser')
            features_loser.append(feature_lose)
        
        # Combinar os datasets
        df_winner = pd.DataFrame(features_winner)
        df_loser = pd.DataFrame(features_loser)
        df = pd.concat([df_winner, df_loser], ignore_index=True)
        
        # Remover linhas com valores nulos
        df = df.dropna()
        
        # Verificar balanceamento
        print(f"\nDistribuição de classes:\n{df['target'].value_counts(normalize=True)}")
        
        return df
    
    def _build_surface_stats(self, matches):
        """Calcula estatísticas por superfície para cada jogador"""
        print("Calculando estatísticas por superfície...")
        self.surface_stats = {}
        
        for surface in ['Clay', 'Hard', 'Grass']:
            surface_matches = matches[matches['surface'] == surface]
            
            # Vitórias por jogador
            winners = surface_matches['winner_name'].value_counts()
            # Derrotas por jogador
            losers = surface_matches['loser_name'].value_counts()
            
            for player in set(winners.index).union(set(losers.index)):
                if player not in self.surface_stats:
                    self.surface_stats[player] = {}
                
                wins = winners.get(player, 0)
                losses = losers.get(player, 0)
                total = wins + losses
                
                self.surface_stats[player][surface] = {
                    'wins': wins,
                    'losses': losses,
                    'win_rate': wins / total if total > 0 else 0.5,
                    'total_matches': total
                }
    
    def _build_h2h_data(self, matches):
        """Constrói o histórico head-to-head entre jogadores"""
        print("Construindo histórico H2H...")
        self.h2h_data = {}
        
        for _, row in matches.iterrows():
            winner = row['winner_name']
            loser = row['loser_name']
            
            # Atualizar registro do vencedor
            if winner not in self.h2h_data:
                self.h2h_data[winner] = {}
            if loser not in self.h2h_data[winner]:
                self.h2h_data[winner][loser] = {'wins': 0, 'losses': 0}
            self.h2h_data[winner][loser]['wins'] += 1
            
            # Atualizar registro do perdedor
            if loser not in self.h2h_data:
                self.h2h_data[loser] = {}
            if winner not in self.h2h_data[loser]:
                self.h2h_data[loser][winner] = {'wins': 0, 'losses': 0}
            self.h2h_data[loser][winner]['losses'] += 1
    
    def _extract_features(self, row, perspective='winner'):
        """Extrai features de uma linha de dados com a perspectiva especificada"""
        if perspective == 'winner':
            player = row['winner_name']
            opponent = row['loser_name']
            player_elo = row['winner_surface_elo']
            opponent_elo = row['loser_surface_elo']
            target = 1
        else:
            player = row['loser_name']
            opponent = row['winner_name']
            player_elo = row['loser_surface_elo']
            opponent_elo = row['winner_surface_elo']
            target = 0
        
        surface = row['surface']
        
        # Head-to-head (do ponto de vista do jogador atual)
        h2h = self.h2h_data.get(player, {}).get(opponent, {'wins': 0, 'losses': 0})
        h2h_wins = h2h['wins']
        h2h_losses = h2h['losses']
        h2h_total = h2h_wins + h2h_losses
        h2h_win_rate = h2h_wins / h2h_total if h2h_total > 0 else 0.5
        
        # Estatísticas por superfície
        player_surface_stats = self.surface_stats.get(player, {}).get(surface, {
            'win_rate': 0.5, 
            'total_matches': 0
        })
        opponent_surface_stats = self.surface_stats.get(opponent, {}).get(surface, {
            'win_rate': 0.5,
            'total_matches': 0
        })
        
        # Features
        features = {
            'elo_diff': player_elo - opponent_elo,
            'player_elo': player_elo,
            'opponent_elo': opponent_elo,
            'h2h_win_rate': h2h_win_rate,
            'h2h_matches': h2h_total,
            'player_surface_win_rate': player_surface_stats['win_rate'],
            'player_surface_matches': player_surface_stats['total_matches'],
            'opponent_surface_win_rate': opponent_surface_stats['win_rate'],
            'opponent_surface_matches': opponent_surface_stats['total_matches'],
            'surface': surface,
            'target': target
        }
        
        # Adicionar features adicionais se disponíveis
        for stat in ['winner_rank', 'loser_rank', 'winner_ht', 'loser_ht', 'winner_age', 'loser_age']:
            if stat in row and pd.notna(row[stat]):
                stat_name = stat.replace('winner_', '').replace('loser_', '')
                if perspective == 'winner':
                    features[f'player_{stat_name}'] = row[stat] if 'winner_' in stat else row[stat.replace('loser_', 'winner_')]
                    features[f'opponent_{stat_name}'] = row[stat] if 'loser_' in stat else row[stat.replace('winner_', 'loser_')]
                else:
                    features[f'player_{stat_name}'] = row[stat] if 'loser_' in stat else row[stat.replace('winner_', 'loser_')]
                    features[f'opponent_{stat_name}'] = row[stat] if 'winner_' in stat else row[stat.replace('loser_', 'winner_')]
        
        return features
    
    def train_model(self, df):
        """Treina o modelo de previsão"""
        print("\nTreinando modelo...")
        
        # Codificar superfície
        df = pd.get_dummies(df, columns=['surface'])
        
        # Separar features e target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Normalizar features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Dividir em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Treinar modelo
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=25,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Calibrar probabilidades
        calibrated_model = CalibratedClassifierCV(self.model, method='isotonic', cv=3)
        calibrated_model.fit(X_train, y_train)
        self.model = calibrated_model
        
        # Avaliar
        y_pred = self.model.predict(X_test)
        print("\nAvaliação do modelo:")
        print(f"Acurácia: {accuracy_score(y_test, y_pred):.3f}")
        print(f"F1-score: {f1_score(y_test, y_pred):.3f}")
        print("\nRelatório de classificação:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            print("\nImportância das features:")
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            print(feature_importance.head(10))
    
    def save_model(self):
        """Salva o modelo e os dados preparados"""
        print("\nSalvando modelo...")
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
            
        joblib.dump(self.model, MODEL_PATH + 'tennis_surface_elo_model.pkl')
        joblib.dump(self.scaler, MODEL_PATH + 'tennis_surface_elo_scaler.pkl')
        joblib.dump({
            'surface_stats': self.surface_stats,
            'h2h_data': self.h2h_data
        }, MODEL_PATH + 'tennis_surface_elo_data.pkl')
        
        print("Modelo e dados salvos com sucesso!")
    
    def load_saved_model(self):
        """Carrega um modelo treinado anteriormente"""
        print("Carregando modelo salvo...")
        self.model = joblib.load(MODEL_PATH + 'tennis_surface_elo_model.pkl')
        self.scaler = joblib.load(MODEL_PATH + 'tennis_surface_elo_scaler.pkl')
        data = joblib.load(MODEL_PATH + 'tennis_surface_elo_data.pkl')
        self.surface_stats = data['surface_stats']
        self.h2h_data = data['h2h_data']
        print("Modelo carregado com sucesso!")
    
    def predict_match(self, player1, player2, surface):
        """Faz uma previsão para uma partida específica com ELO real"""
        # Obter ELOs atualizados por superfície
        player1_elo = self._get_player_elo(player1, surface)
        player2_elo = self._get_player_elo(player2, surface)
        
        # Calcular diferença real
        elo_diff = player1_elo - player2_elo
        
        # Head-to-head
        h2h_wins = self.h2h_data.get(player1, {}).get(player2, {}).get('wins', 0)
        h2h_losses = self.h2h_data.get(player1, {}).get(player2, {}).get('losses', 0)
        h2h_total = h2h_wins + h2h_losses
        h2h_win_rate = h2h_wins / h2h_total if h2h_total > 0 else 0.5
        
        # Estatísticas por superfície
        player1_stats = self.surface_stats.get(player1, {}).get(surface, {'win_rate': 0.5, 'total_matches': 0})
        player2_stats = self.surface_stats.get(player2, {}).get(surface, {'win_rate': 0.5, 'total_matches': 0})
        
        # Preparar features
        features = {
            'elo_diff': elo_diff,
            'player_elo': player1_elo,
            'opponent_elo': player2_elo,
            'h2h_win_rate': h2h_win_rate,
            'h2h_matches': h2h_total,
            'player_surface_win_rate': player1_stats['win_rate'],
            'player_surface_matches': player1_stats['total_matches'],
            'opponent_surface_win_rate': player2_stats['win_rate'],
            'opponent_surface_matches': player2_stats['total_matches'],
            'surface': surface
        }
        
        # Calcular diferença de ELO
        features['elo_diff'] = features['player_elo'] - features['opponent_elo']
        
        # Criar DataFrame e codificar superfície
        df = pd.DataFrame([features])
        df = pd.get_dummies(df, columns=['surface'])
        
        # Garantir que todas as colunas esperadas estão presentes
        expected_columns = set(self.scaler.feature_names_in_)
        missing_cols = expected_columns - set(df.columns)
        for col in missing_cols:
            df[col] = 0
        
        # Reordenar colunas
        df = df[self.scaler.feature_names_in_]
        
        # Normalizar e prever
        X = self.scaler.transform(df)
        prob = self.model.predict_proba(X)[0]
        
        # Resultado
        winner = player1 if prob[1] > 0.5 else player2
        confidence = max(prob[1], prob[0])
        print(f"\nPrevisão: {winner} tem {confidence*100:.1f}% de chance de vencer {player2 if winner == player1 else player1}")
        print(f"Superfície: {surface}")
        print(f"Diferença de ELO: {features['elo_diff']:.1f}")
        print(f"Histórico H2H: {features['h2h_matches']} partidas, taxa de vitória de {player1}: {features['h2h_win_rate']*100:.1f}%")
        print(f"Taxa de vitória em {surface}: {player1} {features['player_surface_win_rate']*100:.1f}%, {player2} {features['opponent_surface_win_rate']*100:.1f}%")
        
        return winner, max(prob)
    
    def _get_player_elo(self, player, surface):
        """Obtém o ELO do jogador para uma superfície específica"""
        # Verifique se os dados estão carregados corretamente
        if not hasattr(self, 'surface_stats'):
            self._build_surface_stats(self.matches)
        
        # Acesse o ELO real do jogador na superfície
        return self.surface_stats.get(player, {}).get(surface, {}).get('elo', INITIAL_ELO)
    
    def _get_h2h_win_rate(self, player1, player2):
        """Calcula a taxa de vitória no head-to-head"""
        stats = self.h2h_data.get(player1, {}).get(player2, {'wins': 0, 'losses': 0})
        total = stats['wins'] + stats['losses']
        return stats['wins'] / total if total > 0 else 0.5
    
    def _get_h2h_matches(self, player1, player2):
        """Obtém o número total de partidas head-to-head"""
        stats = self.h2h_data.get(player1, {}).get(player2, {'wins': 0, 'losses': 0})
        return stats['wins'] + stats['losses']
    
    def _get_surface_win_rate(self, player, surface):
        """Obtém a taxa de vitória do jogador em uma superfície"""
        return self.surface_stats.get(player, {}).get(surface, {}).get('win_rate', 0.5)
    
    def _get_surface_matches(self, player, surface):
        """Obtém o número total de partidas do jogador em uma superfície"""
        return self.surface_stats.get(player, {}).get(surface, {}).get('total_matches', 0)
    
    def interactive_menu(self):
        """Menu interativo para fazer previsões"""
        print("\n=== Tennis Match Predictor ===")
        print("Usando ELO por superfície e histórico H2H")
        
        while True:
            print("\n1. Fazer previsão de partida")
            print("2. Ver exemplos de previsão")
            print("3. Sair")
            choice = input("Escolha: ")
            
            if choice == '1':
                player1 = input("Jogador 1: ").strip()
                player2 = input("Jogador 2: ").strip()
                surface = input("Superfície (Clay/Hard/Grass): ").strip().capitalize()
                
                if surface not in ['Clay', 'Hard', 'Grass']:
                    print("Superfície inválida. Usando Hard como padrão.")
                    surface = 'Hard'
                
                self.predict_match(player1, player2, surface)
            
            elif choice == '2':
                print("\nExemplos de previsão:")
                surfaces = ['Clay', 'Hard', 'Grass']
                players = list(self.surface_stats.keys())
                
                for _ in range(3):
                    p1, p2 = np.random.choice(players, 2, replace=False)
                    surface = np.random.choice(surfaces)
                    self.predict_match(p1, p2, surface)
            
            elif choice == '3':
                print("Saindo...")
                break
            
            else:
                print("Opção inválida")

def main():
    predictor = TennisPredictor()
    
    # Verificar se existe modelo treinado
    if all(os.path.exists(MODEL_PATH + f) for f in [
        'tennis_surface_elo_model.pkl',
        'tennis_surface_elo_scaler.pkl',
        'tennis_surface_elo_data.pkl'
    ]):
        predictor.load_saved_model()
    else:
        predictor.load_data()
        df = predictor.preprocess_data()
        predictor.train_model(df)
        predictor.save_model()
    
    # Menu interativo
    predictor.interactive_menu()

if __name__ == '__main__':
    main()