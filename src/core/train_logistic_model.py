import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

def create_advanced_features(X, feature_names):
    """Cria features avançadas para melhorar o modelo."""
    X_new = X.copy()
    
    # Encontrar índices das features importantes
    int_rate_idx = feature_names.index('int_rate')
    loan_to_income_idx = feature_names.index('loan_to_income')
    annual_inc_idx = feature_names.index('annual_inc')
    dti_idx = feature_names.index('dti')
    
    # 1. Interação entre taxa de juros e loan_to_income
    X_new = np.column_stack([
        X_new,
        X[:, int_rate_idx] * X[:, loan_to_income_idx]
    ])
    
    # 2. Score de risco baseado em múltiplas variáveis
    risk_score = (
        X[:, int_rate_idx] * 0.4 +  # taxa de juros (40% do peso)
        X[:, loan_to_income_idx] * 0.3 +  # razão empréstimo/renda (30% do peso)
        X[:, dti_idx] * 0.3  # razão dívida/renda (30% do peso)
    )
    X_new = np.column_stack([X_new, risk_score])
    
    # 3. Indicador de alto risco (combinação de fatores)
    high_risk = (
        (X[:, int_rate_idx] > X[:, int_rate_idx].mean()) &  # taxa de juros acima da média
        (X[:, loan_to_income_idx] > X[:, loan_to_income_idx].mean()) &  # alto empréstimo/renda
        (X[:, dti_idx] > X[:, dti_idx].mean())  # alto dívida/renda
    ).astype(int)
    X_new = np.column_stack([X_new, high_risk])
    
    # Atualizar lista de features
    new_feature_names = feature_names + [
        'int_rate_x_loan_to_income',
        'risk_score',
        'high_risk_indicator'
    ]
    
    return X_new, new_feature_names

# 1. Carregar os dados preparados
print("1. Carregando dados preparados...")
model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
X_train = np.load(os.path.join(model_dir, 'X_train_transformed.npy'))
X_test = np.load(os.path.join(model_dir, 'X_test_transformed.npy'))
y_train = np.load(os.path.join(model_dir, 'y_train.npy'))
y_test = np.load(os.path.join(model_dir, 'y_test.npy'))

# Carregar nomes das features
with open(os.path.join(model_dir, 'feature_names.pkl'), 'rb') as f:
    feature_names = pickle.load(f)

# 2. Criar features avançadas
print("\n2. Criando features avançadas...")
X_train, new_feature_names = create_advanced_features(X_train, feature_names)
X_test, _ = create_advanced_features(X_test, feature_names)

print("\nNovas features criadas:")
for i, feature in enumerate(new_feature_names):
    if feature not in feature_names:
        print(f"- {feature}")

# 3. Aplicar SMOTE para balancear os dados
print("\n3. Aplicando SMOTE para balancear os dados...")
print("\nDistribuição original das classes:")
print("Classe 0 (Não Default):", sum(y_train == 0))
print("Classe 1 (Default):", sum(y_train == 1))
print(f"Proporção original de Default: {sum(y_train == 1) / len(y_train):.2%}")

# Aplicar SMOTE apenas nos dados de treino
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("\nDistribuição após SMOTE:")
print("Classe 0 (Não Default):", sum(y_train_balanced == 0))
print("Classe 1 (Default):", sum(y_train_balanced == 1))
print(f"Nova proporção de Default: {sum(y_train_balanced == 1) / len(y_train_balanced):.2%}")

# 4. Treinar o modelo
print("\n4. Treinando modelo de Regressão Logística...")
# Ajustando também os hiperparâmetros
model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    C=0.1  # Aumentar regularização para evitar overfitting
)
model.fit(X_train_balanced, y_train_balanced)

# 5. Fazer previsões
print("\n5. Fazendo previsões...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 6. Avaliar o modelo
print("\n6. Avaliando o modelo...")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

print("\nMatriz de Confusão:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

auc_roc = roc_auc_score(y_test, y_pred_proba)
print(f"\nAUC-ROC Score: {auc_roc:.3f}")

# 7. Analisar importância das features
print("\n7. Analisando importância das features...")
feature_importance = pd.DataFrame({
    'feature': new_feature_names,
    'importance': abs(model.coef_[0])
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("\nTop 15 features mais importantes:")
print(feature_importance.head(15))

# 8. Visualizações
print("\n8. Criando visualizações...")

# Matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão')
plt.ylabel('Real')
plt.xlabel('Previsto')
plt.savefig(os.path.join(model_dir, 'confusion_matrix.png'))
plt.close()

# Importância das features
plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
plt.title('Top 15 Features Mais Importantes')
plt.xlabel('Importância Absoluta')
plt.tight_layout()
plt.savefig(os.path.join(model_dir, 'feature_importance.png'))
plt.close()

# 9. Salvar o modelo
print("\n9. Salvando o modelo...")
with open(os.path.join(model_dir, 'logistic_model.pkl'), 'wb') as f:
    pickle.dump(model, f)

# Salvar também os nomes das novas features
with open(os.path.join(model_dir, 'new_feature_names.pkl'), 'wb') as f:
    pickle.dump(new_feature_names, f)

print("\nTreinamento concluído! O modelo está salvo em 'models/logistic_model.pkl'")

# 10. Análise de diferentes pontos de corte
print("\n10. Analisando diferentes pontos de corte...")
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
results = []

for threshold in thresholds:
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_threshold).ravel()
    
    # Calcular métricas
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    results.append({
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1
    })

results_df = pd.DataFrame(results)
print("\nAnálise de diferentes pontos de corte:")
print(results_df.round(3))

# Encontrar o melhor threshold baseado no F1-score
best_threshold = results_df.loc[results_df['f1_score'].idxmax(), 'threshold']
print(f"\nMelhor threshold baseado no F1-score: {best_threshold:.2f}")

# Salvar resultados em CSV
results_df.to_csv(os.path.join(model_dir, 'threshold_analysis.csv'), index=False)
print("\nAnálise de thresholds salva em 'models/threshold_analysis.csv'")
