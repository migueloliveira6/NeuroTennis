import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

MODEL_PATH = 'D:/projetos/Tenis ML-AI/models/'

class TennisPredictor:
    def __init__(self):
        """Inicializa todos os atributos necessários"""
        self.model = None
        self.scaler = None
        self.matches = None
        self.player_history = {}  # Histórico de ELO por jogador
        self.h2h_data = {}        # Dados head-to-head
        self.surface_stats = {}   # Estatísticas por superfície
        self.feature_columns = None  # Colunas usadas no modelo

    def load_data(self):
        """Carrega os dados das partidas"""
        print("Carregando dados de partidas...")
        self.matches = pd.read_csv('wta_matches_with_surface_elo.csv',
                                 parse_dates=['tourney_date'])
        
        # Verificar colunas essenciais
        required_cols = ['winner_name', 'loser_name', 'surface', 'tourney_date',
                        'winner_surface_elo', 'loser_surface_elo']
        if not all(col in self.matches.columns for col in required_cols):
            raise ValueError("Dataset não contém todas as colunas necessárias")
        
        print(f"Dados carregados: {len(self.matches)} partidas")

    def preprocess_data(self):
        """Prepara e processa os dados para treinamento"""
        print("\nPreprocessando dados...")
        
        # Inicializar estruturas de dados
        self.player_history = {}
        self.h2h_data = {}
        self.surface_stats = {}
        
        # Filtrar partidas válidas
        valid_matches = self.matches[
            (self.matches['winner_name'].notna()) & 
            (self.matches['loser_name'].notna()) &
            (self.matches['surface'].notna()) &
            (self.matches['winner_surface_elo'].notna()) &
            (self.matches['loser_surface_elo'].notna())
        ].copy()
        
        # Ordenar por data para processamento temporal
        valid_matches = valid_matches.sort_values('tourney_date')
        
        # Processar cada partida
        features = []
        for _, row in valid_matches.iterrows():
            # Processar estatísticas
            self._process_match_stats(row)
            
            # Obter ELOs antes da partida
            winner_elo_pre = self._get_elo_before_match(row['winner_name'], row['surface'], row['tourney_date'])
            loser_elo_pre = self._get_elo_before_match(row['loser_name'], row['surface'], row['tourney_date'])
            
            if pd.isna(winner_elo_pre) or pd.isna(loser_elo_pre):
                continue
                
            # Perspectiva do vencedor (target=1)
            features.append(self._extract_features(
                row, perspective='winner',
                player_elo=winner_elo_pre,
                opponent_elo=loser_elo_pre,
                target=1
            ))
            
            # Perspectiva do perdedor (target=0)
            features.append(self._extract_features(
                row, perspective='loser',
                player_elo=loser_elo_pre,
                opponent_elo=winner_elo_pre,
                target=0
            ))
        
        # Criar DataFrame de features
        df = pd.DataFrame([f for f in features if f is not None])
        self.feature_columns = df.drop(['target'], axis=1).columns.tolist()
        
        print(f"\nDistribuição de classes:\n{df['target'].value_counts(normalize=True)}")
        return df

    def _process_match_stats(self, row):
        """Processa estatísticas de uma partida e atualiza as estruturas de dados"""
        winner = row['winner_name']
        loser = row['loser_name']
        surface = row['surface']
        date = row['tourney_date']
        
        # Atualizar histórico de jogadores
        for player, is_winner in [(winner, True), (loser, False)]:
            if player not in self.player_history:
                self.player_history[player] = []
            
            self.player_history[player].append({
                'date': date,
                'surface': surface,
                'elo_after': row['winner_surface_elo'] if is_winner else row['loser_surface_elo'],
                'opponent': loser if is_winner else winner,
                'result': 'win' if is_winner else 'loss'
            })
        
        # Atualizar H2H
        if winner not in self.h2h_data:
            self.h2h_data[winner] = {}
        if loser not in self.h2h_data[winner]:
            self.h2h_data[winner][loser] = {'wins': 0, 'losses': 0}
        self.h2h_data[winner][loser]['wins'] += 1
        
        if loser not in self.h2h_data:
            self.h2h_data[loser] = {}
        if winner not in self.h2h_data[loser]:
            self.h2h_data[loser][winner] = {'wins': 0, 'losses': 0}
        self.h2h_data[loser][winner]['losses'] += 1
        
        # Atualizar estatísticas por superfície
        for player, is_winner in [(winner, True), (loser, False)]:
            if player not in self.surface_stats:
                self.surface_stats[player] = {}
            if surface not in self.surface_stats[player]:
                self.surface_stats[player][surface] = {'wins': 0, 'losses': 0}
            
            if is_winner:
                self.surface_stats[player][surface]['wins'] += 1
            else:
                self.surface_stats[player][surface]['losses'] += 1

    def _get_elo_before_match(self, player, surface, date):
        """Obtém o ELO do jogador antes da partida atual"""
        # Para a primeira partida do jogador, usar valor inicial (1500 ou outro)
        if player not in self.player_history:
            return 1500
        
        # Obter o ELO mais recente antes da data atual
        player_matches = self.player_history[player]
        surface_matches = [m for m in player_matches if m['surface'] == surface and m['date'] < date]
        
        if not surface_matches:
            return 1500  # Valor inicial se não houver partidas anteriores nessa superfície
        
        # Retornar o ELO após a última partida (que será o ELO antes da atual)
        return sorted(surface_matches, key=lambda x: x['date'])[-1]['elo_after']

    def _update_player_history(self, row):
        """Atualiza o histórico com os ELOs após esta partida"""
        winner = row['winner_name']
        loser = row['loser_name']
        date = row['tourney_date']
        surface = row['surface']
        
        # Atualizar vencedor
        if winner not in self.player_history:
            self.player_history[winner] = []
        self.player_history[winner].append({
            'date': date,
            'surface': surface,
            'elo_before': self._get_elo_before_match(winner, surface, date),
            'elo_after': row['winner_surface_elo'],
            'opponent': loser,
            'result': 'win'
        })
        
        # Atualizar perdedor
        if loser not in self.player_history:
            self.player_history[loser] = []
        self.player_history[loser].append({
            'date': date,
            'surface': surface,
            'elo_before': self._get_elo_before_match(loser, surface, date),
            'elo_after': row['loser_surface_elo'],
            'opponent': winner,
            'result': 'loss'
        })

    def _extract_features(self, row, perspective, player_elo, opponent_elo, target):
        """Extrai features para uma partida com todos os argumentos necessários"""
        if perspective == 'winner':
            player = row['winner_name']
            opponent = row['loser_name']
        else:
            player = row['loser_name']
            opponent = row['winner_name']
        
        surface = row['surface']
        date = row['tourney_date']
        
        # Head-to-head até antes desta partida
        h2h = self._get_h2h_stats_before_match(player, opponent, date)
        h2h_total = h2h['wins'] + h2h['losses']
        h2h_win_rate = h2h['wins'] / h2h_total if h2h_total > 0 else 0.5
        
        # Estatísticas por superfície até antes desta partida
        player_stats = self._get_surface_stats_before(player, surface, date)
        opponent_stats = self._get_surface_stats_before(opponent, surface, date)
        
        return {
            'date': date,
            'player_elo': player_elo,
            'opponent_elo': opponent_elo,
            'elo_diff': player_elo - opponent_elo,
            'h2h_win_rate': h2h_win_rate,
            'h2h_matches': h2h_total,
            'player_surface_win_rate': player_stats['win_rate'],
            'player_surface_matches': player_stats['total_matches'],
            'opponent_surface_win_rate': opponent_stats['win_rate'],
            'opponent_surface_matches': opponent_stats['total_matches'],
            'surface': surface,
            'target': target
        }

    def _get_h2h_stats_before_match(self, player1, player2, date):
        """Calcula H2H (vitórias e derrotas) até antes de uma data específica"""
        wins = 0
        losses = 0
        
        if player1 not in self.player_history:
            return {'wins': 0, 'losses': 0}
            
        for match in self.player_history[player1]:
            if match['date'] < date and match['opponent'] == player2:
                if match['result'] == 'win':
                    wins += 1
                else:
                    losses += 1
                    
        return {'wins': wins, 'losses': losses}

    def _get_surface_stats_before(self, player, surface, date):
        """Calcula estatísticas por superfície até antes de uma data"""
        wins = losses = 0
        
        if player not in self.player_history:
            return {'win_rate': 0.5, 'total_matches': 0}
            
        for match in self.player_history[player]:
            if match['date'] < date and match['surface'] == surface:
                if match['result'] == 'win':
                    wins += 1
                else:
                    losses += 1
                    
        total = wins + losses
        return {
            'win_rate': wins / total if total > 0 else 0.5,
            'total_matches': total
        }
    
    def _load_historical_data(self):
        """Carrega apenas os dados necessários para previsões"""
        try:
            data = joblib.load(os.path.join(MODEL_PATH, 'tennis_surface_elo_data.pkl'))
            self.player_history = data.get('player_history', {})
            self.h2h_data = data.get('h2h_data', {})
            self.surface_stats = data.get('surface_stats', {})
        except Exception as e:
            print(f"Erro ao carregar dados históricos: {str(e)}")
            self.player_history = {}
            self.h2h_data = {}
            self.surface_stats = {}

    def _get_latest_match_date(self, player1, player2):
        """Obtém a data mais recente em que qualquer um dos jogadores participou"""
        dates = []
        for player in [player1, player2]:
            if player in self.player_history:
                dates.extend([match['date'] for match in self.player_history[player] if match['date']])
        return max(dates) if dates else None

    def train_model(self, df):
        """Treina o modelo com validação temporal"""
        # Ordenar por data
        df = df.sort_values('date')
        
        # Dividir em treino (70%), validação (15%) e teste (15%) temporal
        train_size = int(0.7 * len(df))
        val_size = int(0.15 * len(df))
        
        train = df.iloc[:train_size]
        val = df.iloc[train_size:train_size+val_size]
        test = df.iloc[train_size+val_size:]
        
        # Identificar colunas a serem removidas - apenas as que existem
        cols_to_drop = ['date', 'target']
        for col in ['player', 'opponent']:  # Verifica se essas colunas existem
            if col in df.columns:
                cols_to_drop.append(col)
        
        # Preparar features
        X_train = train.drop(cols_to_drop, axis=1)
        y_train = train['target']
        
        # Codificar superfície
        X_train = pd.get_dummies(X_train, columns=['surface'])
        self.feature_columns = X_train.columns  # Salva as colunas para referência futura
        
        # Treinar modelo
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        self.model.fit(X_train_scaled, y_train)
        
        # Avaliar
        for name, dataset in [('Validação', val), ('Teste', test)]:
            X = dataset.drop(cols_to_drop, axis=1)
            X = pd.get_dummies(X, columns=['surface'])
            
            # Garantir que temos as mesmas colunas que no treino
            X = X.reindex(columns=self.feature_columns, fill_value=0)
            
            X_scaled = self.scaler.transform(X)
            y = dataset['target']
            
            y_pred = self.model.predict(X_scaled)
            print(f"\nAvaliação {name}:")
            print(f"Acurácia: {accuracy_score(y, y_pred):.2f}")
            print(f"F1-score: {f1_score(y, y_pred):.2f}")
            print("Relatório de classificação:")
            print(classification_report(y, y_pred))
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            print("\nImportância das features:")
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            print(feature_importance.head(10))
    
    def predict_match(self, player1, player2, surface, date=None):
        """
        Faz uma previsão para uma partida específica em uma data específica
        
        Args:
            player1 (str): Nome do primeiro jogador
            player2 (str): Nome do segundo jogador
            surface (str): Superfície da partida ('Clay', 'Hard', 'Grass')
            date (datetime): Data da partida (se None, usa a data mais recente)
        
        Returns:
            tuple: (jogador_previsto, probabilidade, detalhes)
        """
        # Verificar se a data é válida
        if date is not None and not isinstance(date, pd.Timestamp):
            try:
                date = pd.to_datetime(date)
            except Exception as e:
                print(f"Erro ao converter data: {e}. Usando data mais recente.")
                date = None

        if date is None:
            date = self.matches['tourney_date'].max()
        
        # Verificar se os jogadores existem no histórico
        if player1 not in self.player_history or player2 not in self.player_history:
            print("Um ou ambos os jogadores não foram encontrados nos dados históricos.")
            return None, None, None
        
        # Obter ELOs antes da partida
        player1_elo = self._get_elo_before_match(player1, surface, date)
        player2_elo = self._get_elo_before_match(player2, surface, date)
        
        # Obter estatísticas até a data
        h2h = self._get_h2h_stats_before_match(player1, player2, date)
        h2h_total = h2h['wins'] + h2h['losses']
        h2h_win_rate = h2h['wins'] / h2h_total if h2h_total > 0 else 0.5
        
        player1_stats = self._get_surface_stats_before(player1, surface, date)
        player2_stats = self._get_surface_stats_before(player2, surface, date)
        
        # Preparar features
        features = {
            'elo_diff': player1_elo - player2_elo,
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
        
        # Criar DataFrame e codificar superfície
        df = pd.DataFrame([features])
        df = pd.get_dummies(df, columns=['surface'])
        
        # Garantir que temos todas as colunas esperadas
        expected_columns = set(self.scaler.feature_names_in_)
        missing_cols = expected_columns - set(df.columns)
        for col in missing_cols:
            df[col] = 0
        
        # Reordenar colunas
        df = df[self.scaler.feature_names_in_]
        
        # Normalizar e prever
        X = self.scaler.transform(df)
        prob = self.model.predict_proba(X)[0]
        
        # Determinar vencedor
        winner = player1 if prob[1] > 0.5 else player2
        confidence = max(prob[1], prob[0])
        
        # Detalhes da previsão
        details = {
            'date': date,
            'surface': surface,
            'player1_elo': player1_elo,
            'player2_elo': player2_elo,
            'elo_diff': features['elo_diff'],
            'h2h': f"{h2h['wins']}-{h2h['losses']}",
            'player1_surface_win_rate': player1_stats['win_rate'],
            'player2_surface_win_rate': player2_stats['win_rate'],
            'probability': confidence
        }
        
        return winner, confidence, details
    
    def save_model(self):
        """Salva o modelo, scaler e dados necessários para previsão futura"""
        print("\nSalvando modelo e dados...")
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        
        # Salvar componentes do modelo
        joblib.dump(self.model, os.path.join(MODEL_PATH, 'tennis_surface_elo_model_wta.pkl'))
        joblib.dump(self.scaler, os.path.join(MODEL_PATH, 'tennis_surface_elo_scaler_wta.pkl'))
        
        # Salvar dados necessários para previsões
        joblib.dump({
            'player_history': self.player_history,
            'h2h_data': self.h2h_data,
            'surface_stats': self.surface_stats,
            'feature_columns': self.feature_columns  # Adicionado para garantir consistência
        }, os.path.join(MODEL_PATH, 'tennis_surface_elo_data_wta.pkl'))
        
        print(f"Modelo e dados salvos em {MODEL_PATH}")

    def load_saved_model(self):
        """Carrega um modelo treinado anteriormente"""
        print("Carregando modelo salvo...")
        try:
            self.model = joblib.load(os.path.join(MODEL_PATH, 'tennis_surface_elo_model_wta.pkl'))
            self.scaler = joblib.load(os.path.join(MODEL_PATH, 'tennis_surface_elo_scaler_wta.pkl'))
            
            data = joblib.load(os.path.join(MODEL_PATH, 'tennis_surface_elo_data_wta.pkl'))
            self.player_history = data['player_history']
            self.h2h_data = data['h2h_data']
            self.surface_stats = data['surface_stats']
            self.feature_columns = data['feature_columns']
            
            print("Modelo e dados carregados com sucesso!")
            return True
        except Exception as e:
            print(f"Erro ao carregar modelo salvo: {str(e)}")
            return False

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
    
    def main_menu(self):
        """Menu interativo para fazer previsões"""
        print("\n=== Tennis Match Predictor ===")
        print("Usando ELO dinâmico por superfície")
        
        while True:
            print("\nMenu Principal:")
            print("1. Fazer previsão de partida")
            print("2. Ver histórico de um jogador")
            print("3. Sair")
            
            choice = input("Escolha uma opção: ").strip()
            
            if choice == '1':
                self._prediction_menu()
            elif choice == '2':
                self._player_history_menu()
            elif choice == '3':
                print("Saindo do programa...")
                break
            else:
                print("Opção inválida. Tente novamente.")

    def _prediction_menu(self):
        """Submenu para fazer previsões"""
        print("\n--- Previsão de Partida ---")
        
        # Verificar se os dados necessários estão carregados
        if not hasattr(self, 'player_history') or not self.player_history:
            print("Carregando dados históricos...")
            self._load_historical_data()  # Método que carrega apenas os dados necessários
        
        player1 = input("Nome do Jogador 1: ").strip()
        player2 = input("Nome do Jogador 2: ").strip()
        surface = input("Superfície (Clay/Hard/Grass): ").strip().capitalize()
        
        # Validar superfície
        if surface not in ['Clay', 'Hard', 'Grass']:
            print("Superfície inválida. Usando 'Hard' como padrão.")
            surface = 'Hard'
        
        date_input = input("Data da partida (YYYY-MM-DD, deixe vazio para data mais recente): ").strip()
        
        try:
            date = pd.to_datetime(date_input) if date_input else None
        except:
            print("Formato de data inválido. Usando data mais recente.")
            date = None
        
        # Determinar data mais recente se não foi especificada
        if date is None:
            date = self._get_latest_match_date(player1, player2)
            if date is None:
                print("Não foi possível determinar a data mais recente para estes jogadores.")
                return
        
        # Fazer previsão
        winner, confidence, details = self.predict_match(player1, player2, surface, date)
        
        if winner:
            print("\n=== Resultado da Previsão ===")
            date_str = date.strftime('%Y-%m-%d') if date else 'Data mais recente'
            print(f"Data: {date_str}")
            print(f"Superfície: {surface}")
            print(f"\nProbabilidade: {winner} tem {confidence*100:.1f}% de chance de vencer")
            print(f"Diferença de ELO: {details['elo_diff']:.1f}")
            
            # Formatar histórico H2H
            h2h_text = "Igual (sem confrontos anteriores)" if details['h2h'] == "0-0" else f"{details['h2h']} (Vitórias de {player1})"
            print(f"Histórico H2H: {h2h_text}")
            
            print(f"Taxa de vitória em {surface}:")
            print(f"- {player1}: {details['player1_surface_win_rate']*100:.1f}%")
            print(f"- {player2}: {details['player2_surface_win_rate']*100:.1f}%")
        else:
            print("Não foi possível fazer a previsão.")

    def _player_history_menu(self):
        """Submenu para visualizar histórico de jogador"""
        print("\n--- Histórico de Jogador ---")
        
        player = input("Nome do Jogador: ").strip()
        
        if player not in self.player_history:
            print("Jogador não encontrado nos dados históricos.")
            return
        
        # Mostrar estatísticas gerais
        print(f"\nEstatísticas para {player}:")
        
        # Por superfície
        surfaces = set(m['surface'] for m in self.player_history[player])
        for surface in surfaces:
            stats = self._get_surface_stats_before(player, surface, datetime.now())
            print(f"\n{surface}:")
            print(f"Partidas: {stats['total_matches']}")
            print(f"Taxa de vitória: {stats['win_rate']*100:.1f}%")
        
        # Últimos ELOs
        print("\nÚltimos ELOs por superfície:")
        for surface in surfaces:
            last_match = next(
                (m for m in reversed(self.player_history[player]) 
                 if m['surface'] == surface), None)
            if last_match:
                print(f"{surface}: {last_match['elo_after']:.1f}")

def main():
    predictor = TennisPredictor()
    
    # Verificar se existe modelo treinado
    if all(os.path.exists(os.path.join(MODEL_PATH, f)) for f in [
        'tennis_surface_elo_model_wta.pkl',
        'tennis_surface_elo_scaler_wta.pkl',
        'tennis_surface_elo_data_wta.pkl'
    ]):
        print("Modelo treinado encontrado. Carregando...")
        if not predictor.load_saved_model():
            print("Falha ao carregar modelo. Treinando novo modelo...")
            predictor.load_data()
            df = predictor.preprocess_data()
            predictor.train_model(df)
            predictor.save_model()
    else:
        print("Nenhum modelo treinado encontrado. Treinando novo modelo...")
        predictor.load_data()
        df = predictor.preprocess_data()
        predictor.train_model(df)
        predictor.save_model()
    
    predictor.main_menu()

if __name__ == '__main__':
    main()