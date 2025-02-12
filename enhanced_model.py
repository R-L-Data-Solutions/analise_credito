import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import pickle
import os
from analysis_interest_rate import calcular_taxa_juros

# Simular um banco de dados com histórico
def criar_dados_historicos(n_samples=1000):
    np.random.seed(42)
    
    dados = {
        # Features básicas (que já tínhamos)
        'loan_amnt': np.random.uniform(1000, 40000, n_samples),
        'int_rate': np.random.uniform(5, 25, n_samples),
        'annual_inc': np.random.uniform(30000, 150000, n_samples),
        
        # NOVO: Histórico de Pagamentos
        'dias_atraso_media': np.random.uniform(0, 30, n_samples),
        'parcelas_pagas_pontualmente': np.random.uniform(0, 1, n_samples),  # % de parcelas em dia
        'maior_atraso': np.random.uniform(0, 90, n_samples),
        
        # NOVO: Relacionamento com Banco
        'tempo_conta_anos': np.random.uniform(0, 20, n_samples),
        'saldo_medio': np.random.uniform(-1000, 50000, n_samples),
        'usa_cheque_especial': np.random.choice([0, 1], n_samples),
        
        # NOVO: Score de Crédito
        'score_credito': np.random.uniform(0, 1000, n_samples),
        'consultas_cpf_6m': np.random.randint(0, 10, n_samples),
        'tem_outros_emprestimos': np.random.choice([0, 1], n_samples)
    }
    
    # Criar target (default) baseado em regras realistas
    probabilidade_default = (
        0.3 * (dados['dias_atraso_media'] > 5) +  # Atrasos frequentes
        0.2 * (dados['parcelas_pagas_pontualmente'] < 0.8) +  # Histórico ruim
        0.15 * (dados['score_credito'] < 600) +  # Score baixo
        0.15 * (dados['usa_cheque_especial'] == 1) +  # Usa cheque especial
        0.1 * (dados['consultas_cpf_6m'] > 3) +  # Muitas consultas
        0.1 * (dados['tem_outros_emprestimos'] == 1)  # Tem outros empréstimos
    )
    
    dados['default'] = np.random.binomial(1, probabilidade_default)
    
    return pd.DataFrame(dados)

def treinar_modelo_avancado():
    """Treina um modelo com features avançadas de histórico."""
    print("1. Criando dados de treinamento com histórico...")
    df = criar_dados_historicos()
    
    # Separar features e target
    X = df.drop('default', axis=1)
    y = df['default']
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\n2. Distribuição de Default nos dados:")
    print(f"Taxa de Default: {y.mean():.1%}")
    
    # Treinar modelo
    print("\n3. Treinando modelo com features de histórico...")
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Avaliar modelo
    print("\n4. Avaliando o modelo:")
    y_pred = model.predict(X_test)
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))
    
    # Analisar importância das features
    print("\n5. Features mais importantes:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': abs(model.coef_[0])
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 features mais importantes:")
    for _, row in feature_importance.head(10).iterrows():
        print(f"{row['feature']}: {row['importance']:.3f}")
    
    return model, X.columns

def analisar_novo_caso(model, feature_names, dados_cliente):
    """
    Analisa um novo caso com dados de histórico.
    
    Exemplo de dados_cliente:
    {
        # Dados básicos
        'loan_amnt': 15000,
        'int_rate': 12.5,
        'annual_inc': 60000,
        
        # Histórico de Pagamentos
        'dias_atraso_media': 2,
        'parcelas_pagas_pontualmente': 0.95,
        'maior_atraso': 15,
        
        # Relacionamento com Banco
        'tempo_conta_anos': 5,
        'saldo_medio': 8000,
        'usa_cheque_especial': 0,
        
        # Score de Crédito
        'score_credito': 750,
        'consultas_cpf_6m': 2,
        'tem_outros_emprestimos': 0
    }
    """
    # Preparar dados no formato correto
    X = np.zeros(len(feature_names))
    for i, feature in enumerate(feature_names):
        X[i] = dados_cliente.get(feature, 0)
    
    # Fazer previsão
    X = X.reshape(1, -1)
    prob_default = model.predict_proba(X)[0][1]
    is_default = model.predict(X)[0]
    
    # Analisar fatores de risco
    fatores_risco = []
    if dados_cliente['dias_atraso_media'] > 5:
        fatores_risco.append(f"Média de {dados_cliente['dias_atraso_media']:.1f} dias de atraso")
    if dados_cliente['parcelas_pagas_pontualmente'] < 0.8:
        fatores_risco.append(f"Apenas {dados_cliente['parcelas_pagas_pontualmente']*100:.1f}% das parcelas pagas em dia")
    if dados_cliente['score_credito'] < 600:
        fatores_risco.append(f"Score de crédito baixo: {dados_cliente['score_credito']:.0f}")
    if dados_cliente['usa_cheque_especial'] == 1:
        fatores_risco.append("Utiliza cheque especial")
    if dados_cliente['consultas_cpf_6m'] > 3:
        fatores_risco.append(f"{dados_cliente['consultas_cpf_6m']} consultas ao CPF nos últimos 6 meses")
    
    return {
        'probabilidade_default': prob_default * 100,
        'previsao': 'ALTO RISCO' if is_default == 1 else 'BAIXO RISCO',
        'fatores_risco': fatores_risco
    }

if __name__ == "__main__":
    # 1. Treinar modelo
    model, feature_names = treinar_modelo_avancado()
    
    # 2. Exemplo de um caso de baixo risco
    print("\n=== EXEMPLO: CLIENTE BAIXO RISCO ===")
    cliente_bom = {
        'loan_amnt': 15000,
        'int_rate': 12.5,
        'annual_inc': 80000,
        'dias_atraso_media': 0,
        'parcelas_pagas_pontualmente': 0.98,
        'maior_atraso': 0,
        'tempo_conta_anos': 8,
        'saldo_medio': 15000,
        'usa_cheque_especial': 0,
        'score_credito': 850,
        'consultas_cpf_6m': 1,
        'tem_outros_emprestimos': 0
    }
    
    resultado = analisar_novo_caso(model, feature_names, cliente_bom)
    print(f"\nPrevisão: {resultado['previsao']}")
    print(f"Probabilidade de Default: {resultado['probabilidade_default']:.1f}%")
    if resultado['fatores_risco']:
        print("\nFatores de Risco Identificados:")
        for fator in resultado['fatores_risco']:
            print(f"- {fator}")
    else:
        print("\nNenhum fator de risco significativo identificado!")
    
    # 3. Exemplo de um caso de alto risco
    print("\n=== EXEMPLO: CLIENTE ALTO RISCO ===")
    cliente_ruim = {
        'loan_amnt': 25000,
        'int_rate': 18.5,
        'annual_inc': 45000,
        'dias_atraso_media': 8,
        'parcelas_pagas_pontualmente': 0.7,
        'maior_atraso': 45,
        'tempo_conta_anos': 1,
        'saldo_medio': -500,
        'usa_cheque_especial': 1,
        'score_credito': 520,
        'consultas_cpf_6m': 5,
        'tem_outros_emprestimos': 1
    }
    
    resultado = analisar_novo_caso(model, feature_names, cliente_ruim)
    print(f"\nPrevisão: {resultado['previsao']}")
    print(f"Probabilidade de Default: {resultado['probabilidade_default']:.1f}%")
    if resultado['fatores_risco']:
        print("\nFatores de Risco Identificados:")
        for fator in resultado['fatores_risco']:
            print(f"- {fator}")
    else:
        print("\nNenhum fator de risco significativo identificado!")
