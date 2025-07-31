<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body>

<h1>NeuroTennis an (Tennis-ML/AI)</h1>

<h2>Descrição</h2>
<p>
  Este projeto aplica <strong>Machine Learning</strong> e <strong>Inteligência Artificial</strong> para prever o vencedor de partidas de tênis com base em dados históricos e estatísticas de jogadores. Além disso, implementa um sistema de <strong>ratings Elo</strong> para avaliar a performance relativa dos jogadores, com suporte a <strong>visualizações temporais</strong> da evolução desses ratings.
</p>
<p>
  O objetivo é fornecer uma ferramenta robusta e interpretável para análise de desempenho, combinando previsão de partidas e sistemas de ranqueamento.
</p>

<h2>Funcionalidades</h2>
<ul>
  <li><strong>🔮 Previsão de Vencedores:</strong> modelo de ML treinado com histórico de desempenho e confrontos diretos.</li>
  <li><strong>📈 Cálculo de Elo Ratings:</strong> ranqueamento dinâmico baseado em resultados reais.</li>
  <li><strong>📊 Visualização Temporal:</strong> gráficos da evolução dos ratings Elo dos jogadores.</li>
</ul>

<h2>Tecnologias Utilizadas</h2>
<ul>
  <li>Python</li>
  <li>Scikit-learn</li>
  <li>Pandas & NumPy</li>
  <li>Matplotlib & Seaborn</li>
  <li>Jupyter Notebooks</li>
</ul>

<h2>📁 Estrutura do Repositório</h2>
<pre><code>Tennis-ML-AI/
├── data/                # Conjuntos de dados
├── figures/             # Gráficos gerados
├── models/              # Modelos salvos
├── notebooks/           # Notebooks Jupyter
├── src/
│   ├── model_elo_v2.py  # Script de previsão
│   ├── visualization.py # Geração de gráficos
├── requirements.txt     # Dependências
└── README.md            # Este arquivo</code></pre>

<h2>Como Usar</h2>

<h3>1. Clonar o Repositório</h3>
<pre><code>git clone https://github.com/migueloliveira6/Tennis-ML-AI.git
cd Tennis-ML-AI</code></pre>

<h3>2. Instalar Dependências</h3>
<pre><code>python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
pip install -r requirements.txt</code></pre>

<h3>3. Executar o Projeto</h3>
<p><strong>Treinar modelo:</strong></p>
<pre><code>python src/model_elo_xgboost.py</code></pre>

<p><strong>Gerar visualizações:</strong></p>
<pre><code> notebooks/atp_chall_itf_matches_elo.ipynb</code></pre>
<a> <img src="https://github.com/migueloliveira6/NeuroTennis/blob/main/notebooks/dashboard%20rafa%20nadal.png"/> </a>

<h3>4. Explorar Notebooks</h3>
<p>Explore os notebooks em <code>notebooks/</code> para análises e exemplos.</p>

<h2>Conjunto de Dados</h2>
<p>
  Inclui resultados de partidas, estatísticas de jogadores e informações de torneios.
</p>
<p><em>Nota: Inclua as fontes ou instruções de obtenção dos dados, se necessário.</em></p>

<h2>Exemplos de Resultados</h2>
<ul>
  <li><strong>Previsão:</strong> Acurracy do <code>72%</code> na previsão.</li>
  <li><strong>Rankings Elo:</strong> Ranking atualizado dos Top 10 jogadores.</li>
  <li><strong>Visualizações:</strong> Exemplos disponíveis na pasta <code>figures/</code>.</li>
</ul>

<h2>🤝 Contribuições</h2>
<ol>
  <li>Faça um fork do repositório</li>
  <li>Crie uma nova branch: <code>git checkout -b feature/nova-funcionalidade</code></li>
  <li>Commit suas alterações: <code>git commit -m "Adiciona nova funcionalidade"</code></li>
  <li>Envie um pull request</li>
</ol>

<h2>📬 Contato</h2>
<p>Para dúvidas ou sugestões, entre em contato com <strong>Miguel Oliveira</strong>:<br>
<a href="https://github.com/migueloliveira6">github.com/migueloliveira6</a></p>

</body>
</html>
