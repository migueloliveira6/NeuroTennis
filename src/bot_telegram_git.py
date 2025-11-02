from dotenv import load_dotenv
import os
import sys
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

print("🔄 Carregando imports...")
load_dotenv()

TOKEN_BOT = os.getenv('TOKEN_BOT')
CHAT_ID = os.getenv('CHAT_ID')
MODEL_PATH = os.getenv('MODEL_PATH')
PREVISOES_PATH = os.getenv('PREVISOES_PATH')
NAME_LOOKUP = os.getenv('NAME_LOOKUP', 'name_lookup.csv')
if not TOKEN_BOT:
    print("⚠️ TOKEN_BOT não encontrado — verifica se está definido nos Secrets do GitHub.")
print("done")

# Configuração de logging para Jupyter
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("✅ Imports carregados com sucesso!")

nome_cache = {}

def getPlayersFullName(playerUrl: str) -> str:
    """
    Função otimizada para obter nome completo do jogador
    
    Args:
        playerUrl: URL relativa do jogador
        
    Returns:
        Nome completo normalizado do jogador
    """
    # Verificar cache primeiro
    if playerUrl in nome_cache:
        return nome_cache[playerUrl]
    
    try:
        player_url = "https://www.tennisexplorer.com" + playerUrl
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        player_response = requests.get(player_url, headers=headers, timeout=10)
        player_response.raise_for_status()
        
        player_soup = BeautifulSoup(player_response.content, "html.parser")
        player_table = player_soup.find("table", {"class": "plDetail"})
        
        if not player_table:
            nome_cache[playerUrl] = "Unknown"
            return "Unknown"
            
        player_table_body = player_table.find("tbody")
        
        # Extrair nome
        player_name = player_table_body.find_all("h3")
        if not player_name:
            nome_cache[playerUrl] = "Unknown"
            return "Unknown"
            
        name = " ".join(player_name[0].text.split())
        splitname = name.split(" ")
        
        if len(splitname) >= 2:
            first_name = splitname[-1]
            last_name = name.replace(" " + first_name, "")
            name = first_name + " " + last_name
        
        name = name.strip().replace("-", " ")
        
        # Aplicar substituições personalizadas do CSV se existir
        try:
            name_dict = pd.read_csv({NAME_LOOKUP})
            for _, item in name_dict.iterrows():
                name = name.replace(item.old, item.new)
        except FileNotFoundError:
            logger.debug("Arquivo name_lookup.csv não encontrado - usando nomes originais")
        except Exception as e:
            logger.debug(f"Erro ao aplicar substituições de nome: {e}")
        
        # Armazenar no cache
        nome_cache[playerUrl] = name
        return name
        
    except Exception as e:
        logger.warning(f"Erro ao extrair nome do jogador {playerUrl}: {e}")
        nome_cache[playerUrl] = "Unknown"
        return "Unknown"

print("✅ Função getPlayersFullName carregada!")

@dataclass
class Jogo:
    """Classe para representar um jogo de tênis"""
    player1: str
    player2: str
    odd1: Optional[float]
    odd2: Optional[float]
    
    def __str__(self):
        odds_info = f" (Odds: {self.odd1} vs {self.odd2})" if self.odd1 and self.odd2 else ""
        return f"🎾 {self.player1} vs {self.player2}{odds_info}"

print("✅ Dataclass Jogo definida!")

class NeuroTennis:
    """Classe otimizada para scraping do Tennis Explorer"""
    
    # Constantes compiladas para melhor performance
    BLACKLIST_TORNEIO = frozenset({
        "itf", "utr", "junior", "exhibition", "boodles tennis challenge", 
        "boodles tennis cup", "boodles tennis series", "series", "cup"
        "pro", "match", "finals", "world university games", "challenger"
    })
    
    BLACKLIST_NOMES = frozenset({
        "challenger", "itf", "wta", "utr", "series", "pro", "match", "cup", 
        "junior", "wimbledon", "french", "us open", "australian", "masters", 
        "finals", "exhibition", "eastbourne", "miami", "indian wells", "paris", 
        "rome", "shanghai", "tokyo", "beijing", "doha", "dubai", "sydney", 
        "montreal", "cincinnati", "stuttgart", "birmingham", "brussels", 
        "istanbul", "moscow", "st petersburg", "basel", "rotterdam", 
        "hamburg", "vienna", "london"
    })
    
    # Headers otimizado
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "pt-BR,pt;q=0.8,en;q=0.5,en-US;q=0.3",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Cache-Control": "no-cache"
    }
    
    def __init__(self, timeout: int = 10):
        """
        Inicializa o scraper
        
        Args:
            timeout: Timeout para requisições HTTP
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        logger.info("🚀 NeuroTennis inicializado!")
    
    def _is_atp_torneio(self, nome: str) -> bool:
        """Verifica se é um torneio ATP válido"""
        if not nome:
            return False
        
        nome_lower = nome.lower()
        return not any(blacklisted in nome_lower for blacklisted in self.BLACKLIST_TORNEIO)
    
    def _is_nome_valido(self, nome: str) -> bool:
        """Verifica se o nome do jogador é válido """
        if not nome or len(nome.split()) > 5:
            return False
        
        nome_lower = nome.lower()
        return not any(blocked in nome_lower for blocked in self.BLACKLIST_NOMES)
    
    def _limpar_nome_jogador(self, nome_bruto: str) -> str:
        """
        Limpa o nome do jogador removendo seeds, iniciais e outros caracteres extras
        
        Args:
            nome_bruto: Nome bruto extraído do HTML
            
        Returns:
            Nome limpo para validação
        """
        if not nome_bruto:
            return ""
        
        # Remover seeds de torneio: (1), (6), etc
        nome = re.sub(r'\([^)]*\)', '', nome_bruto)
        
        # Remover iniciais: "B.", "J.", etc
        nome = re.sub(r'\b[A-Z]\.\s*', '', nome)
        
        # Remover espaços extras
        nome = ' '.join(nome.split())
        
        return nome.strip()
    
    def _extrair_odds(self, linha) -> Tuple[Optional[float], Optional[float]]:
        """Extrai as odds de uma linha"""
        try:
            odds_elements = linha.find_all("td", class_=["course", "coursew"])
            if len(odds_elements) < 2:
                return None, None
            
            # Verifica qual elemento tem a classe 'coursew' para determinar a ordem
            if "coursew" in odds_elements[0].get("class", []):
                odd1 = float(odds_elements[0].get_text(strip=True))
                odd2 = float(odds_elements[1].get_text(strip=True))
            elif "coursew" in odds_elements[1].get("class", []):
                odd1 = float(odds_elements[1].get_text(strip=True))
                odd2 = float(odds_elements[0].get_text(strip=True))
            else:
                odd1 = float(odds_elements[0].get_text(strip=True))
                odd2 = float(odds_elements[1].get_text(strip=True))
            
            return odd1, odd2
            
        except (ValueError, AttributeError, IndexError) as e:
            logger.debug(f"Erro ao extrair odds: {e}")
            return None, None
    
    def _processar_par_jogadores(self, linha1, linha2, torneio_atual: str) -> Optional[Jogo]:
        """Processa um par de linhas para extrair informações dos jogadores"""
        try:
            # Extrair elementos dos jogadores
            jogador1_td = linha1.find("td", class_="t-name")
            jogador2_td = linha2.find("td", class_="t-name")
            
            if not all([jogador1_td, jogador2_td, 
                       jogador1_td.a, jogador2_td.a]):
                return None
            
            # Extrair informações básicas
            href1 = jogador1_td.a.get("href", "")
            href2 = jogador2_td.a.get("href", "")
            nome_bruto1 = jogador1_td.get_text(strip=True)
            nome_limpo1 = self._limpar_nome_jogador(nome_bruto1)
            nome_bruto2 = jogador2_td.get_text(strip=True)
            nome_limpo2 = self._limpar_nome_jogador(nome_bruto2)
            
            # Validar nomes
            if not (self._is_nome_valido(nome_limpo1) and 
                   self._is_nome_valido(nome_limpo2)):
                return None
            
            # Extrair odds
            odd1, odd2 = self._extrair_odds(linha1)
            
            # Obter nomes completos dos jogadores usando a função original
            jogador1 = getPlayersFullName(href1)
            jogador2 = getPlayersFullName(href2)
            
            return Jogo(
                player1=jogador1,
                player2=jogador2,
                odd1=odd1,
                odd2=odd2
            )
            
        except Exception as e:
            logger.warning(f"Erro ao processar par de jogadores: {e}")
            return None
        
    def _processar_par_jogadores_debug(self, linha1, linha2, torneio_atual: str) -> Optional[Jogo]:
        """Versão com debug para identificar problemas"""
        try:
            # Log das linhas sendo processadas
            jogador1_td = linha1.find("td", class_="t-name")
            jogador2_td = linha2.find("td", class_="t-name")
            
            # Debug: Mostrar o que foi encontrado
            if jogador1_td and jogador1_td.a:
                href1 = jogador1_td.a.get("href", "")
                nome1 = jogador1_td.get_text(strip=True)
                print(f"🔍 Jogador 1: {nome1} ({href1})")
            else:
                print(f"❌ Linha 1 não tem jogador válido")
                return None
                
            if jogador2_td and jogador2_td.a:
                href2 = jogador2_td.a.get("href", "")
                nome2 = jogador2_td.get_text(strip=True)
                print(f"🔍 Jogador 2: {nome2} ({href2})")
            else:
                print(f"❌ Linha 2 não tem jogador válido")
                return None
            
            # Validar nomes
            if not (self._is_nome_valido(nome1) and self._is_nome_valido(nome2)):
                print(f"❌ Nomes inválidos: {nome1} / {nome2}")
                return None
            
            # Extrair odds
            odd1, odd2 = self._extrair_odds(linha1)
            
            # Obter nomes completos
            jogador1 = getPlayersFullName(href1)
            jogador2 = getPlayersFullName(href2)
            
            print(f"✅ Par válido: {jogador1} vs {jogador2}")
            
            return Jogo(
                player1=jogador1,
                player2=jogador2,
                odd1=odd1,
                odd2=odd2
            )
            
        except Exception as e:
            logger.warning(f"⚠️ Erro ao processar par: {e}")
            return None
        
    def _construir_url(self, dias_offset: int = 1) -> str:
        """Constrói a URL para scraping"""
        hoje = datetime.today()
        data_target = hoje + timedelta(days=dias_offset)
        
        return (f"https://www.tennisexplorer.com/matches/"
                f"?type=atp-single&year={data_target.year}"
                f"&month={data_target.month:02d}"
                f"&day={data_target.day:02d}")
    
    def _extrair_nome_torneio(self, linha) -> Optional[str]:
        """Extrai o nome do torneio de uma linha de cabeçalho"""
        try:
            td_nome = linha.find("td", class_="t-name")
            if td_nome and td_nome.a:
                nome_torneio = td_nome.a.get_text(strip=True)
                
                # Filtros específicos
                if nome_torneio == "Tampere challenger":
                    return None
                
                if self._is_atp_torneio(nome_torneio):
                    return nome_torneio
            
            return None
            
        except Exception as e:
            logger.debug(f"Erro ao extrair nome do torneio: {e}")
            return None
    
    def extrair_jogos_agrupados_por_torneio(self, dias_offset: int = 1) -> Dict[str, List[Jogo]]:
        """
        Extrai jogos agrupados por torneio de forma otimizada
        
        Args:
            dias_offset: Dias a partir de hoje para buscar jogos
            
        Returns:
            Dicionário com torneios e seus respectivos jogos
        """
        url = self._construir_url(dias_offset)
        logger.info(f"📅 URL sendo acessada: {url}")
        
        try:
            # Fazer requisição
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, "html.parser")
            tabelas = soup.select("table.result")
            
            if not tabelas:
                logger.warning("⚠️ Nenhuma tabela encontrada.")
                return {}
            
            torneios = {}
            torneio_atual = None
            
            # Processar cada tabela
            for tabela in tabelas:
                linhas = tabela.select("tr")
                i = 0
                
                while i < len(linhas):
                    linha = linhas[i]
                    classes_linha = linha.get("class", [])
                    
                    # Detectar cabeçalho de torneio
                    if "head" in classes_linha and "flags" in classes_linha:
                        torneio_atual = self._extrair_nome_torneio(linha)
                        if torneio_atual:
                            torneios.setdefault(torneio_atual, [])
                            logger.info(f"🏟️ Torneio encontrado: {torneio_atual}")
                        i += 1
                        continue
                    
                    # ✅ CORREÇÃO: Verificar se AMBAS as linhas têm dados de jogadores
                    if torneio_atual and i + 1 < len(linhas):
                        linha_atual = linhas[i]
                        linha_seguinte = linhas[i + 1]
                        
                        # Verificar se ambas têm a classe "t-name" (indicador de jogador)
                        tem_jogador_atual = linha_atual.find("td", class_="t-name") is not None
                        tem_jogador_seguinte = linha_seguinte.find("td", class_="t-name") is not None
                        
                        # Verificar se não são cabeçalhos
                        classes_atual = linha_atual.get("class", [])
                        classes_seguinte = linha_seguinte.get("class", [])
                        eh_header_atual = "head" in classes_atual or "flags" in classes_atual
                        eh_header_seguinte = "head" in classes_seguinte or "flags" in classes_seguinte
                        
                        # Só processar se AMBAS forem linhas de jogadores válidas
                        if (tem_jogador_atual and tem_jogador_seguinte and 
                            not eh_header_atual and not eh_header_seguinte):
                            
                            jogo = self._processar_par_jogadores(
                                linha_atual, linha_seguinte, torneio_atual
                            )
                            
                            if jogo:
                                torneios[torneio_atual].append(jogo)
                                i += 2  # ✅ Avançar 2 linhas (par processado)
                            else:
                                # Se falhou mas eram linhas válidas, pode ser erro nos dados
                                # Tentar avançar 2 na mesma para não desalinhar
                                i += 2
                        else:
                            # Linha atual não é jogador válido, avançar só 1
                            i += 1
                    else:
                        i += 1
            
            # Remover torneios vazios
            torneios = {k: v for k, v in torneios.items() if v}
            
            logger.info(f"✅ Total de torneios encontrados: {len(torneios)}")
            return torneios
            
        except requests.RequestException as e:
            logger.error(f"❌ Erro na requisição HTTP: {e}")
            return {}
        except Exception as e:
            logger.error(f"❌ Erro inesperado: {e}")
            return {}
    
    def __del__(self):
        """Limpa recursos ao destruir o objeto"""
        if hasattr(self, 'session'):
            self.session.close()

print("✅ Classe NeuroTennis definida!")

# Função de conveniência para compatibilidade com código existente
def extrair_jogos_agrupados_por_torneio(dias_offset: int = 1) -> Dict[str, List[Dict]]:
    """
    Função wrapper para manter compatibilidade com código existente
    
    Args:
        dias_offset: Dias a partir de hoje
        
    Returns:
        Dicionário com jogos no formato original
    """
    scraper = NeuroTennis()
    
    try:
        jogos = scraper.extrair_jogos_agrupados_por_torneio(dias_offset)
        
        # Converter para formato original (dict ao invés de dataclass)
        resultado = {}
        for torneio, lista_jogos in jogos.items():
            resultado[torneio] = [
                {
                    "player1": jogo.player1,
                    "player2": jogo.player2,
                    "odd1": jogo.odd1,
                    "odd2": jogo.odd2
                }
                for jogo in lista_jogos
            ]
        
        return resultado
        
    finally:
        del scraper

# Função legada para compatibilidade total
def is_atp_torneio(nome):
    """Função legada - usar NeuroTennis._is_atp_torneio"""
    scraper = NeuroTennis()
    return scraper._is_atp_torneio(nome)

def is_nome_valido(nome):
    """Função legada - usar NeuroTennis._is_nome_valido"""
    scraper = NeuroTennis()
    return scraper._is_nome_valido(nome)

print("✅ Funções de compatibilidade definidas!")

# ## 🎯 Exemplo de Uso - Método Orientado a Objetos (Recomendado)

# Criar instância do scraper
scraper = NeuroTennis(timeout=15)

# Extrair jogos de hoje
print("🔄 Extraindo jogos...")
jogos_hoje = scraper.extrair_jogos_agrupados_por_torneio(dias_offset=1)

print(f"\n📊 Resumo da extração:")
print(f"Total de torneios encontrados: {len(jogos_hoje)}")

# Mostrar resultados detalhados
total_jogos = 0
for torneio, jogos in jogos_hoje.items():
    total_jogos += len(jogos)
    print(f"\n🏟️ {torneio}: {len(jogos)} jogos")
    for jogo in jogos:
        print(f"  {jogo}")

print(f"\n🎾 Total de jogos extraídos: {total_jogos}")

# ## 🔄 Exemplo de Uso - Função Legada (Para Compatibilidade)

# Usar a função wrapper para compatibilidade com código existente
print("🔄 Usando função legada...")
jogos_compativel = extrair_jogos_agrupados_por_torneio(dias_offset=1)

print(f"📊 Formato compatível: {len(jogos_compativel)} torneios encontrados")
print("Exemplo do primeiro torneio:")
if jogos_compativel:
    primeiro_torneio = list(jogos_compativel.keys())[0]
    primeiros_jogos = jogos_compativel[primeiro_torneio][:2]  # Primeiros 2 jogos
    
    print(f"\n🏟️ {primeiro_torneio}:")
    for jogo in primeiros_jogos:
        print(f"  🎾 {jogo['player1']} vs {jogo['player2']}")
        if jogo['odd1'] and jogo['odd2']:
            print(f"     Odds: {jogo['odd1']} vs {jogo['odd2']}")

# ## 📈 Análise de Performance e Cache

# Verificar status do cache de nomes
print(f"📋 Cache de nomes: {len(nome_cache)} entradas")
if nome_cache:
    print("Primeiras 5 entradas do cache:")
    for i, (url, nome) in enumerate(list(nome_cache.items())[:5]):
        print(f"  {i+1}. {url.split('/')[-1]} -> {nome}")

# Função para limpar cache se necessário
def limpar_cache():
    """Limpa o cache de nomes"""
    global nome_cache
    nome_cache.clear()
    print("🧹 Cache limpo!")

# Uncomment para limpar cache se necessário
# limpar_cache()

def estatisticas_torneios(jogos_dict: Dict):
    """
    Mostra estatísticas dos torneios extraídos
    
    Args:
        jogos_dict: Dicionário de jogos
    """
    if not jogos_dict:
        print("❌ Nenhum jogo para analisar")
        return
    
    print("\n📈 Estatísticas dos Torneios:")
    print("-" * 50)
    
    total_jogos = sum(len(jogos) for jogos in jogos_dict.values())
    jogos_com_odds = sum(
        1 for jogos in jogos_dict.values() 
        for jogo in jogos 
        if (isinstance(jogo, dict) and jogo.get('odd1') and jogo.get('odd2')) or
           (hasattr(jogo, 'odd1') and jogo.odd1 and jogo.odd2)
    )
    
    print(f"🏟️ Total de torneios: {len(jogos_dict)}")
    print(f"🎾 Total de jogos: {total_jogos}")
    print(f"💰 Jogos com odds: {jogos_com_odds}")
    print(f"📊 Média de jogos por torneio: {total_jogos/len(jogos_dict):.1f}")
    
    # Top 5 torneios com mais jogos
    torneios_ordenados = sorted(jogos_dict.items(), key=lambda x: len(x[1]), reverse=True)
    print(f"\n🏆 Top 5 torneios com mais jogos:")
    for i, (torneio, jogos) in enumerate(torneios_ordenados[:5]):
        print(f"  {i+1}. {torneio}: {len(jogos)} jogos")

# Exemplo de uso das funções auxiliares
if jogos_hoje:
    estatisticas_torneios(jogos_hoje)

print("\n✅ Notebook carregado com sucesso! Pronto para uso.")

print("Total de torneios encontrados:", len(jogos_hoje))
print("Jogos extraídos com sucesso!")
print(jogos_hoje)

for torneio, jogos in jogos_hoje.items():
    # Elimina jogos sem odds
    jogos_validos = [
        j for j in jogos
        if j.odd1 is not None and j.odd2 is not None
    ]

    for j in jogos_validos:
        if j.player1.startswith(" "):
            j.player1 = j.player1.lstrip()
        if j.player2.startswith(" "):
            j.player2 = j.player2.lstrip()

    if not jogos_validos:
        continue  # pula torneios sem jogos válidos

    print(f"\n🎾 Torneio: {torneio}")
    for j in jogos_validos:
        print(f"  {j.player1} vs {j.player2} — Odds: {j.odd1} / {j.odd2}")

print("✅ Jogos filtrados e exibidos com sucesso!")

# Caminho absoluto até à pasta src
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(BASE_DIR, "src")

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from model_elo_xgboost import TennisPredictor

predictor = TennisPredictor()

model_files = [
    'tennis_surface_elo_model_xgboost.pkl',
    'tennis_surface_elo_scaler_xgboost.pkl',
    'tennis_surface_elo_data_xgboost.pkl'
]

if all(os.path.exists(os.path.join(MODEL_PATH, f)) for f in model_files):
    print("✅ Modelo encontrado. Carregando...")
    if not predictor.load_saved_model():
        print("⚠️ Falha ao carregar. Treinando novo...")
        predictor.load_data()
        df = predictor.preprocess_data()
        predictor.train_model(df)
        predictor.save_model()
else:
    print("⚠️ Nenhum modelo encontrado. Treinando novo...")
    predictor.load_data()
    df = predictor.preprocess_data()
    predictor.train_model(df)
    predictor.save_model()

print("✅ Modelo pronto para previsões!")

@dataclass
class ResultadoPrevisao:
    """Classe para representar resultado de uma previsão"""
    torneio: str
    jogador1: str
    jogador2: str
    vencedor_previsto: str
    confianca: float
    elo_diff: float
    h2h: str
    odd1: Optional[float]
    odd2: Optional[float]
    superficie: str
    valor_aposta: Optional[float] = None
    roi_esperado: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário"""
        return {
            "Torneio": self.torneio,
            "Jogador 1": self.jogador1,
            "Jogador 2": self.jogador2,
            "Vencedor Previsto": self.vencedor_previsto,
            "Confiança (%)": round(self.confianca * 100, 1),
            "ELO Diff": round(self.elo_diff, 1),
            "H2H": self.h2h,
            "Odd 1": self.odd1,
            "Odd 2": self.odd2,
            "Superfície": self.superficie,
            "Valor Aposta": round(self.valor_aposta, 3) if self.valor_aposta is not None and not pd.isna(self.valor_aposta) else 0,
            "ROI Esperado (%)": round(self.roi_esperado * 100, 1) if self.roi_esperado is not None and not pd.isna(self.roi_esperado) else 0
        }

print("✅ Estruturas de dados definidas!")

class SuperficieDetector:
    """Classe para detectar a superfície do torneio de forma inteligente"""

    # Dicionários organizados por superfície com padrões mais específicos
    SUPERFICIE_PATTERNS = {
        "Grass": {
            # Torneios específicos de relva
            "torneios": {
                "wimbledon", "queen's club", "queens club", "halle", "stuttgart open",
                "eastbourne", "mallorca", "newport", "birmingham", "nottingham",
                "hertogenbosch", "libema open", "nature valley"
            },
            # Padrões que indicam relva
            "patterns": [
                r"grass\s*(court|championship)?",
                r"lawn\s*tennis",
                r"queen'?s?\s*club",
                r"pre.*wimbledon",
                r"wimbledon.*warm.*up"
            ]
        },

        "Hard": {
            # Torneios específicos de hard
            "torneios": {
                "us open", "indian wells", "miami", "shanghai", "beijing", "tokyo",
                "toronto", "montreal", "cincinnati", "washington", "atlanta",
                "los cabos", "san diego", "winston salem", "new york", "flushing",
                "australian open", "atp finals", "masters cup", "davis cup finals",
                "laver cup", "hopman cup", "atp cup", "united cup", "doha", "dubai",
                "chengdu", "hangzhou", "almaty", "stockholm", "brussels", "metz",
                "rotterdam", "basel", "vienna", "paris", "athens", "sydney"
            },
            # Padrões que indicam hard
            "patterns": [
                r"hard\s*(court)?",
                r"us\s*open",
                r"australian\s*open",
                r"indian\s*wells",
                r"miami\s*(open|masters)?",
                r"masters\s*(1000|series)",
                r"atp\s*(finals|cup)",
                r"indoor.*hard"
            ]
        },

        "Clay": {
            # Torneios específicos de clay
            "torneios": {
                "french open", "roland garros", "monte carlo", "rome", "madrid",
                "barcelona", "hamburg", "munich", "estoril", "geneva", "lyon",
                "bucharest", "budapest", "umag", "gstaad", "bastad", "kitzbuhel",
                "casablanca", "marrakech", "houston", "charleston", "bogota"
            },
            # Padrões que indicam clay
            "patterns": [
                r"clay\s*(court)?",
                r"french\s*open",
                r"roland\s*garros",
                r"monte\s*carlo",
                r"tierra\s*batida",
                r"red\s*clay",
                r"outdoor.*clay"
            ]
        }
    }

    # Cache para evitar reprocessamento
    _cache = {}

    @classmethod
    def detectar_superficie(cls, nome_torneio: str, data: datetime = None) -> str:
        """
        Detecta a superfície do torneio de forma inteligente

        Args:
            nome_torneio: Nome do torneio
            data: Data do torneio (para contexto sazonal)

        Returns:
            Superfície detectada: "Grass", "Hard", ou "Clay"
        """
        if not nome_torneio:
            return "Hard"  # Default

        # Verificar cache
        cache_key = nome_torneio.lower().strip()
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        nome_lower = nome_torneio.lower()
        superficie_detectada = cls._detectar_por_nome_e_patterns(nome_lower)

        # Se não conseguiu detectar, usar contexto sazonal
        if not superficie_detectada and data:
            superficie_detectada = cls._detectar_por_sazonalidade(data)

        # Default para Hard se ainda não detectou
        superficie_final = superficie_detectada or "Hard"

        # Armazenar no cache
        cls._cache[cache_key] = superficie_final

        return superficie_final

    @classmethod
    def _detectar_por_nome_e_patterns(cls, nome_lower: str) -> Optional[str]:
        """Detecta superfície por nome e padrões"""
        for superficie, config in cls.SUPERFICIE_PATTERNS.items():
            # Verificar torneios específicos
            for torneio in config["torneios"]:
                if torneio in nome_lower:
                    return superficie

            # Verificar padrões regex
            for pattern in config["patterns"]:
                if re.search(pattern, nome_lower, re.IGNORECASE):
                    return superficie

        return None

    @classmethod
    def _detectar_por_sazonalidade(cls, data: datetime) -> Optional[str]:
        """Detecta superfície baseada na sazonalidade"""
        mes = data.month

        # Temporada de grama (maio-julho)
        if mes in [5, 6, 7]:
            return "Grass"

        # Temporada de saibro europeu (abril-junho, setembro-outubro)
        if mes in [4, 5]:
            return "Clay"

        # Resto do ano tende a ser Hard Court
        return "Hard"

    @classmethod
    def limpar_cache(cls):
        """Limpa o cache de superfícies"""
        cls._cache.clear()
        logger.info("🧹 Cache de superfícies limpo!")

print("✅ SuperficieDetector definido!")

class TennisPredicaoAnalyzer:
    """Classe principal para análise de previsões de tênis"""

    def __init__(self, predictor, max_workers: int = 4, calcular_roi: bool = True):
        """
        Inicializa o analisador

        Args:
            predictor: Instância do seu modelo de previsão
            max_workers: Número máximo de threads para processamento paralelo
            calcular_roi: Se deve calcular ROI esperado
        """
        self.predictor = predictor
        self.max_workers = max_workers
        self.calcular_roi = calcular_roi
        self.superficie_detector = SuperficieDetector()

        logger.info(f"🔮 NeuroTennis iniciado!")
        logger.info(f"   - Processamento paralelo: {max_workers} workers")
        logger.info(f"   - Cálculo de ROI: {'Ativado' if calcular_roi else 'Desativado'}")

    def _filtrar_jogos_validos(self, jogos_dict: Dict) -> List[Tuple[str, Dict]]:
        """
        Filtra jogos válidos (com odds) para análise

        Args:
            jogos_dict: Dicionário de jogos por torneio

        Returns:
            Lista de tuplas (torneio, jogo) válidas
        """
        jogos_validos = []

        for torneio, jogos in jogos_dict.items():
            for jogo in jogos:
                # Verificar se tem odds válidas
                if isinstance(jogo, dict):
                    odd1 = jogo.get("odd1")
                    odd2 = jogo.get("odd2")
                else:  # Dataclass
                    odd1 = getattr(jogo, 'odd1', None)
                    odd2 = getattr(jogo, 'odd2', None)

                if odd1 is not None and odd2 is not None:
                    jogos_validos.append((torneio, jogo))

        return jogos_validos

    def _processar_jogo_individual(self, torneio: str, jogo: Any, data: datetime) -> Optional[ResultadoPrevisao]:
        """
        Processa um jogo individual

        Args:
            torneio: Nome do torneio
            jogo: Dados do jogo (dict ou dataclass)
            data: Data de referência

        Returns:
            ResultadoPrevisao ou None se erro
        """
        try:
            # Extrair dados do jogo
            if isinstance(jogo, dict):
                p1 = jogo.get("player1")
                p2 = jogo.get("player2")
                odd1 = jogo.get("odd1")
                odd2 = jogo.get("odd2")
            else:  # Dataclass
                p1 = getattr(jogo, 'player1', None)
                p2 = getattr(jogo, 'player2', None)
                odd1 = getattr(jogo, 'odd1', None)
                odd2 = getattr(jogo, 'odd2', None)

            if not all([p1, p2, odd1, odd2]):
                return None

            # Detectar superfície
            superficie = self.superficie_detector.detectar_superficie(torneio, data)

            # Fazer previsão
            winner, prob, detalhes = self.predictor.predict_match(p1, p2, superficie, date=data)

            # Calcular métricas de aposta se solicitado
            valor_aposta = None
            roi_esperado = None

            if self.calcular_roi:
                valor_aposta, roi_esperado = self._calcular_metricas_aposta(
                    winner, prob, p1, p2, odd1, odd2
                )

            return ResultadoPrevisao(
                torneio=torneio,
                jogador1=p1,
                jogador2=p2,
                vencedor_previsto=winner,
                confianca=prob,
                elo_diff=detalhes.get("elo_diff", 0),
                h2h=detalhes.get("h2h", "N/A"),
                odd1=odd1,
                odd2=odd2,
                superficie=superficie,
                valor_aposta=valor_aposta,
                roi_esperado=roi_esperado
            )

        except Exception as e:
            logger.warning(f"⚠️ Erro ao processar {p1} vs {p2}: {e}")
            return None

    def _calcular_metricas_aposta(self, winner: str, prob: float, p1: str, p2: str,
                                 odd1: float, odd2: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Calcula métricas de aposta (valor e ROI esperado)

        Args:
            winner: Jogador previsto como vencedor
            prob: Probabilidade de vitória
            p1, p2: Nomes dos jogadores
            odd1, odd2: Odds dos jogadores

        Returns:
            Tupla (valor_da_aposta, roi_esperado)
        """
        try:
            # Determinar qual odd usar baseado no vencedor previsto
            if winner == p1:
                odd_vencedor = odd1
            else:
                odd_vencedor = odd2

            # Verificar se odd é válida
            if not odd_vencedor or odd_vencedor <= 0:
                return 0, None

            # Calcular probabilidade implícita da casa de apostas
            prob_implicita = 1 / odd_vencedor

            # Calcular valor da aposta (Kelly Criterion simplificado)
            if prob > prob_implicita and odd_vencedor > 1:
                valor_aposta = (prob * odd_vencedor - 1) / (odd_vencedor - 1)
                roi_esperado = prob * (odd_vencedor - 1) - (1 - prob)

                # Verificar se os valores são válidos (não NaN ou infinito)
                if pd.isna(valor_aposta) or pd.isna(roi_esperado):
                    return 0, 0

                return max(0, min(round(valor_aposta, 3), 1)), round(roi_esperado, 3)

            return 0, 0

        except (ZeroDivisionError, TypeError, ValueError):
            return 0, 0

    def analisar_jogos(self, jogos_dict: Dict, data: datetime = None,
                      processar_paralelo: bool = True) -> Tuple[List[ResultadoPrevisao], Dict[str, Any]]:
        """
        Analisa todos os jogos e gera previsões

        Args:
            jogos_dict: Dicionário de jogos por torneio
            data: Data de referência (default: hoje)
            processar_paralelo: Se deve usar processamento paralelo

        Returns:
            Tupla (lista_resultados, estatisticas)
        """
        if data is None:
            data = datetime.today()

        logger.info("🔄 Iniciando análise de jogos...")

        # Filtrar jogos válidos
        jogos_validos = self._filtrar_jogos_validos(jogos_dict)

        logger.info(f"📊 Jogos encontrados:")
        logger.info(f"   - Total de torneios: {len(jogos_dict)}")
        logger.info(f"   - Jogos válidos (com odds): {len(jogos_validos)}")

        if not jogos_validos:
            logger.warning("⚠️ Nenhum jogo válido encontrado!")
            return [], {}

        # Processar jogos
        resultados = []

        if processar_paralelo and len(jogos_validos) > 1:
            resultados = self._processar_paralelo(jogos_validos, data)
        else:
            resultados = self._processar_sequencial(jogos_validos, data)

        # Filtrar resultados válidos
        resultados_validos = [r for r in resultados if r is not None]

        # Gerar estatísticas
        estatisticas = self._gerar_estatisticas(resultados_validos, jogos_dict)

        logger.info(f"✅ Análise concluída:")
        logger.info(f"   - Previsões geradas: {len(resultados_validos)}")
        logger.info(f"   - Sucessos: {estatisticas['previsoes_geradas']}")
        logger.info(f"   - Erros: {estatisticas['erros']}")

        return resultados_validos, estatisticas

    def _processar_paralelo(self, jogos_validos: List, data: datetime) -> List[Optional[ResultadoPrevisao]]:
        """Processa jogos em paralelo"""
        logger.info(f"⚡ Processando {len(jogos_validos)} jogos em paralelo...")

        resultados = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submeter todas as tarefas
            future_to_jogo = {
                executor.submit(self._processar_jogo_individual, torneio, jogo, data): (torneio, jogo)
                for torneio, jogo in jogos_validos
            }

            # Coletar resultados conforme completam
            for future in as_completed(future_to_jogo):
                try:
                    resultado = future.result(timeout=30)  # 30s timeout por jogo
                    resultados.append(resultado)
                except Exception as e:
                    torneio, jogo = future_to_jogo[future]
                    logger.warning(f"⚠️ Erro no processamento paralelo: {e}")
                    resultados.append(None)

        return resultados

    def _processar_sequencial(self, jogos_validos: List, data: datetime) -> List[Optional[ResultadoPrevisao]]:
        """Processa jogos sequencialmente"""
        logger.info(f"🔄 Processando {len(jogos_validos)} jogos sequencialmente...")

        resultados = []

        for i, (torneio, jogo) in enumerate(jogos_validos, 1):
            if i % 10 == 0:  # Log a cada 10 jogos
                logger.info(f"   Processando jogo {i}/{len(jogos_validos)}...")

            resultado = self._processar_jogo_individual(torneio, jogo, data)
            resultados.append(resultado)

        return resultados

    def _gerar_estatisticas(self, resultados: List[ResultadoPrevisao], jogos_dict: Dict) -> Dict[str, Any]:
        """Gera estatísticas da análise"""
        if not resultados:
            return {"previsoes_geradas": 0, "erros": 0}

        total_jogos = sum(len(jogos) for jogos in jogos_dict.values())
        total_jogos_validos = len([r for r in resultados if r])

        # Estatísticas por superfície
        superficie_stats = {}
        for resultado in resultados:
            if resultado:
                sup = resultado.superficie
                if sup not in superficie_stats:
                    superficie_stats[sup] = {"count": 0, "confianca_media": 0}
                superficie_stats[sup]["count"] += 1
                superficie_stats[sup]["confianca_media"] += resultado.confianca

        # Calcular médias
        for sup_data in superficie_stats.values():
            if sup_data["count"] > 0:
                sup_data["confianca_media"] /= sup_data["count"]

        # Estatísticas de ROI se disponível
        roi_stats = {}
        if self.calcular_roi:
            apostas_recomendadas = [r for r in resultados if r and r.valor_aposta and r.valor_aposta > 0]
            if apostas_recomendadas:
                roi_stats = {
                    "apostas_recomendadas": len(apostas_recomendadas),
                    "roi_medio": sum(r.roi_esperado for r in apostas_recomendadas) / len(apostas_recomendadas),
                    "valor_medio_aposta": sum(r.valor_aposta for r in apostas_recomendadas) / len(apostas_recomendadas)
                }

        return {
            "total_torneios": len(jogos_dict),
            "total_jogos": total_jogos,
            "jogos_validos": total_jogos_validos,
            "previsoes_geradas": len(resultados),
            "erros": total_jogos - total_jogos_validos,
            "superficie_stats": superficie_stats,
            "roi_stats": roi_stats
        }

    def gerar_dataframe(self, resultados: List[ResultadoPrevisao]) -> pd.DataFrame:
        """
        Converte resultados para DataFrame pandas

        Args:
            resultados: Lista de resultados

        Returns:
            DataFrame com os resultados
        """
        if not resultados:
            return pd.DataFrame()

        data = [resultado.to_dict() for resultado in resultados]
        df = pd.DataFrame(data)

        # Ordenar por confiança decrescente
        if "Confiança (%)" in df.columns:
            df = df.sort_values("Confiança (%)", ascending=False)

        return df.reset_index(drop=True)

    def salvar_resultados(self, resultados: List[ResultadoPrevisao],
                         caminho: str = None, formato: str = "csv") -> str:
        """
        Salva resultados em arquivo

        Args:
            resultados: Lista de resultados
            caminho: Caminho do arquivo (auto-gerado se None)
            formato: Formato do arquivo ("csv", "excel", "json")

        Returns:
            Caminho do arquivo salvo
        """
        if not resultados:
            raise ValueError("Nenhum resultado para salvar")

        if caminho is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            caminho = f"previsoes_tenis_{timestamp}.{formato}"

        df = self.gerar_dataframe(resultados)

        if formato.lower() == "csv":
            df.to_csv(caminho, index=False, encoding='utf-8')
        elif formato.lower() in ["excel", "xlsx"]:
            df.to_excel(caminho, index=False)
        elif formato.lower() == "json":
            df.to_json(caminho, orient='records', indent=2)
        else:
            raise ValueError(f"Formato não suportado: {formato}")

        logger.info(f"💾 Resultados salvos em: {caminho}")
        return caminho

print("✅ TennisPredicaoAnalyzer definido!")

# ## 🎯 Exemplo de Uso Completo

# analyzer = TennisPredicaoAnalyzer(predictor, max_workers=4, calcular_roi=True)

# ## 🔧 Função Otimizada de Análise (Drop-in Replacement)

def analisar_jogos_otimizado(jogos_hoje: Dict, predictor, processar_paralelo: bool = True,
                           calcular_roi: bool = True, salvar_arquivo: str = None) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Função otimizada para análise de jogos - substituto direto da função original

    Args:
        jogos_hoje: Dicionário de jogos extraídos
        predictor: Instância do modelo de previsão
        processar_paralelo: Se deve usar processamento paralelo
        calcular_roi: Se deve calcular métricas de ROI
        salvar_arquivo: Caminho para salvar resultados (opcional)

    Returns:
        Tupla (lista_resultados_dict, dataframe)
    """
    # Criar analisador
    analyzer = TennisPredicaoAnalyzer(
        predictor=predictor,
        max_workers=4,
        calcular_roi=calcular_roi
    )

    # Analisar jogos
    resultados, estatisticas = analyzer.analisar_jogos(
        jogos_hoje,
        processar_paralelo=processar_paralelo
    )

    # Converter para formato compatível (lista de dicts)
    resultados_dict = [resultado.to_dict() for resultado in resultados]

    # Gerar DataFrame
    df = analyzer.gerar_dataframe(resultados)

    # Salvar se solicitado
    if salvar_arquivo:
        analyzer.salvar_resultados(resultados, salvar_arquivo)

    # Imprimir estatísticas
    print("\n📊 Estatísticas da Análise:")
    print("-" * 50)
    print(f"🏟️ Total de torneios: {estatisticas['total_torneios']}")
    print(f"🎾 Total de jogos: {estatisticas['total_jogos']}")
    print(f"✅ Jogos válidos: {estatisticas['jogos_validos']}")
    print(f"🔮 Previsões geradas: {estatisticas['previsoes_geradas']}")
    print(f"❌ Erros: {estatisticas['erros']}")

    if estatisticas.get('superficie_stats'):
        print(f"\n🏟️ Por Superfície:")
        for superficie, dados in estatisticas['superficie_stats'].items():
            print(f"   - {superficie}: {dados['count']} jogos (confiança média: {dados['confianca_media']:.1%})")

    if estatisticas.get('roi_stats'):
        roi_stats = estatisticas['roi_stats']
        print(f"\n💰 Estatísticas de ROI:")
        print(f"   - Apostas recomendadas: {roi_stats.get('apostas_recomendadas', 0)}")
        print(f"   - ROI médio esperado: {roi_stats.get('roi_medio', 0):.1%}")
        print(f"   - Valor médio de aposta: {roi_stats.get('valor_medio_aposta', 0):.1%}")

    return resultados_dict, df

print("✅ Função de análise otimizada definida!")

# Analisar com a função otimizada
resultados, df = analisar_jogos_otimizado(
    jogos_hoje=jogos_hoje,
    predictor=predictor,  # Substitua pela sua instância
    processar_paralelo=True,
    calcular_roi=True,
)

df_resultados = pd.DataFrame(resultados)
df_resultados = df_resultados.sort_values(by="Confiança (%)", ascending=False).reset_index(drop=True)
print("\nResultados dos jogos previstos:")
print(df_resultados)

# Salva o arquivo com a data de hoje no nome
csv_path = os.path.join(PREVISOES_PATH, "previsoes_tenis.csv")
df_resultados.to_csv(csv_path, index=False)
print(f"📁 Previsões salvas em: {csv_path}")
print("✅ Scraping e previsões concluídos com sucesso!")

# Enviar notificação via Telegram

MENSAGEM = "🎾 Previsões de Ténis prontas para amanhã!"

url = f"https://api.telegram.org/bot{TOKEN_BOT}/sendMessage"
res = requests.post(url, data={"chat_id": CHAT_ID, "text": MENSAGEM})
print("✅ Enviado:", res.json())

df = pd.read_csv(csv_path)

linhas = []
for _, row in df.iterrows():
    valor_aposta = round(row['Valor Aposta'], 3) if pd.notna(row['Valor Aposta']) else 'N/A'
    roi_esperado = round(row['ROI Esperado (%)'], 3) if pd.notna(row['ROI Esperado (%)']) else 'N/A'
    linha = (
        f"Jogo: {row['Jogador 1']} vs {row['Jogador 2']}\n"
        f"🎯 Previsto: {row['Vencedor Previsto']} ({row['Confiança (%)']}%)\n"
        f"ELO Diff: {row['ELO Diff']} | H2H: {row['H2H']}\n"
        f"Valor Aposta: {valor_aposta}\n"
        f"ROI Esperado: {roi_esperado}\n"
        f"Odds: {row['Odd 1']} / {row['Odd 2']}\n"
        f"Torneio: {row['Torneio']}\n"
        f"──────────────"
    )
    linhas.append(linha)

mensagem = "\n".join(linhas)

print("📩 Enviando mensagem formatada...")
print("Mensagem pronta para envio:")
print(mensagem)

url = f"https://api.telegram.org/bot{TOKEN_BOT}/sendMessage"
res = requests.post(url, data={
    "chat_id": CHAT_ID,
    "text": mensagem,
})

print("✅ Enviado para Telegram:", res.status_code)

if res.status_code == 200:
    print("Mensagem enviada com sucesso!✅")
else:
    # Divide a lista de linhas em duas partes
    meio = len(linhas) // 2
    mensagem1 = "\n".join(linhas[:meio])
    mensagem2 = "\n".join(linhas[meio:])

    print("Mensagem 1:")
    print(mensagem1)
    print("\nMensagem 2:")
    print(mensagem2)
    # Envia a primeira parte
    res1 = requests.post(url, data={
        "chat_id": CHAT_ID,
        "text": mensagem1,
    })
    print("✅ Enviado primeira parte:", res1.status_code)
    # Envia a segunda parte
    res2 = requests.post(url, data={
        "chat_id": CHAT_ID,
        "text": mensagem2,
    })
    print("✅ Enviado segunda parte:", res2.status_code)
    if res1.status_code == 200 and res2.status_code == 200:
        print("Mensagens enviadas com sucesso!")
    else:
        print("⚠️ Falha ao enviar mensagens:", res1.status_code, res2.status_code)
        print("Tentando dividir em 3 partes...")
        total_linhas = len(linhas)
        terco = total_linhas // 3

        # Cálculo dos índices para divisão em 3 partes
        primeira_parte = terco
        segunda_parte = terco * 2

        mensagem1 = "\n".join(linhas[:primeira_parte])
        mensagem2 = "\n".join(linhas[primeira_parte:segunda_parte])
        mensagem3 = "\n".join(linhas[segunda_parte:])
        # Envia a primeira parte
        res1 = requests.post(url, data={
            "chat_id": CHAT_ID,
            "text": mensagem1,
        })
        # Envia a segunda parte
        res2 = requests.post(url, data={
            "chat_id": CHAT_ID,
            "text": mensagem2,
        })
        # Envia a terceira parte
        res3 = requests.post(url, data={
            "chat_id": CHAT_ID,
            "text": mensagem3,
        })
        print("✅ Enviado primeira parte:", res1.status_code)
        print("✅ Enviado segunda parte:", res2.status_code)
        print("✅ Enviado terceira parte:", res3.status_code)
        # Verifica se todas as mensagens foram enviadas com sucesso
        if res1.status_code == 200 and res2.status_code == 200 and res3.status_code == 200:
            print("Mensagens enviadas com sucesso!")
        else:
            print("⚠️ Falha ao enviar mensagens:", res1.status_code, res2.status_code, res3.status_code)