import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from train_model2 import predict_match

# Carregar o modelo classificador e de regressão
model_classification = joblib.load('D:/projetos/Tenis ML-AI/models/modelo_treinado.pkl')
scaler = joblib.load('D:/projetos/Tenis ML-AI/models/scaler_treinado.pkl')
matches, df_encoded, rank_df, h2h_data, surface_stats = joblib.load('D:/projetos/Tenis ML-AI/models/dados_preparados.pkl')

# Interface do usuário com Streamlit
st.title('Previsão de Jogos de Tênis')

p1 = st.text_input('Nome do jogador 1', 'Daniil Medvedev')
p2 = st.text_input('Nome do jogador 2', 'Reilly Opelka')
surface = st.selectbox('Escolha a superfície', ['Clay', 'Hard', 'Grass'])

if st.button('Fazer previsão'):
    # Usando a função importada para fazer a previsão de vencedor
    winner, confidence = predict_match(p1, p2, surface, model_classification, scaler, rank_df, h2h_data, surface_stats)
    st.write(f'O vencedor previsto é: {winner} com {confidence * 100:.2f}% de confiança.')