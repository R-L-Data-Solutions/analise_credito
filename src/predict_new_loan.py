import pickle
import numpy as np
import pandas as pd
import os

def carregar_modelo():
    """Carrega o modelo treinado e os nomes das features."""
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    
    # Carregar o modelo
    with open(os.path.join(model_dir, 'logistic_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    
    # Carregar nomes das features
    with open(os.path.join(model_dir, 'feature_names.pkl'), 'rb') as f:
        feature_names = pickle.load(f)
    
    return model, feature_names

def analisar_emprestimo(dados_emprestimo):
    """
    Analisa um novo pedido de empréstimo.
    
    Exemplo de uso:
    dados = {
        'loan_amnt': 10000,        # Valor do empréstimo
        'int_rate': 12.5,          # Taxa de juros
        'annual_inc': 50000,       # Renda anual
        'dti': 15.5,               # Razão dívida/renda
        'loan_to_income': 0.2,     # Razão empréstimo/renda
        'grade': 'B',              # Nota de crédito
        'purpose': 'debt_consolidation'  # Finalidade
    }
    """
    # Carregar modelo e features
    model, feature_names = carregar_modelo()
    
    # Preparar os dados no formato correto
    X = np.zeros(len(feature_names))
    
    # Preencher valores numéricos
    numeric_fields = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'loan_to_income']
    for field in numeric_fields:
        if field in dados_emprestimo and field in feature_names:
            idx = feature_names.index(field)
            X[idx] = dados_emprestimo[field]
    
    # Preencher campos categóricos
    if 'grade' in dados_emprestimo:
        grade = dados_emprestimo['grade']
        grade_feature = f'grade_{grade}'
        if grade_feature in feature_names:
            idx = feature_names.index(grade_feature)
            X[idx] = 1
    
    if 'purpose' in dados_emprestimo:
        purpose = dados_emprestimo['purpose']
        purpose_feature = f'purpose_{purpose}'
        if purpose_feature in feature_names:
            idx = feature_names.index(purpose_feature)
            X[idx] = 1
    
    # Fazer a previsão
    X = X.reshape(1, -1)
    prob_default = model.predict_proba(X)[0][1]
    is_default = model.predict(X)[0]
    
    # Preparar o resultado
    resultado = {
        'probabilidade_default': round(prob_default * 100, 1),
        'previsao': 'ALTO RISCO' if is_default == 1 else 'BAIXO RISCO',
        'nivel_confianca': 'ALTA' if abs(prob_default - 0.5) > 0.3 else 'MÉDIA' if abs(prob_default - 0.5) > 0.15 else 'BAIXA'
    }
    
    return resultado

def imprimir_resultado(resultado):
    """Imprime o resultado da análise de forma clara."""
    print("\n=== RESULTADO DA ANÁLISE DE RISCO ===")
    print(f"Previsão: {resultado['previsao']}")
    print(f"Probabilidade de Default: {resultado['probabilidade_default']}%")
    print(f"Nível de Confiança: {resultado['nivel_confianca']}")
    print("=====================================")

# Exemplo de uso
if __name__ == "__main__":
    # Exemplo 1: Empréstimo de baixo risco
    print("\nANALISANDO EMPRÉSTIMO DE BAIXO RISCO:")
    dados_baixo_risco = {
        'loan_amnt': 10000,
        'int_rate': 8.5,
        'annual_inc': 80000,
        'dti': 12.5,
        'loan_to_income': 0.125,
        'grade': 'A',
        'purpose': 'debt_consolidation'
    }
    resultado = analisar_emprestimo(dados_baixo_risco)
    imprimir_resultado(resultado)
    
    # Exemplo 2: Empréstimo de alto risco
    print("\nANALISANDO EMPRÉSTIMO DE ALTO RISCO:")
    dados_alto_risco = {
        'loan_amnt': 35000,
        'int_rate': 18.5,
        'annual_inc': 45000,
        'dti': 28.5,
        'loan_to_income': 0.78,
        'grade': 'E',
        'purpose': 'small_business'
    }
    resultado = analisar_emprestimo(dados_alto_risco)
    imprimir_resultado(resultado)
