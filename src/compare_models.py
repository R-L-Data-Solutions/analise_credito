import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

def avaliar_modelo(y_true, y_pred, y_pred_proba, nome_modelo):
    """Função para avaliar e mostrar resultados do modelo de forma clara."""
    print(f"\n{'='*20} Avaliação do Modelo: {nome_modelo} {'='*20}")
    
    # 1. Métricas básicas
    print("\n1. Métricas Principais:")
    print(classification_report(y_true, y_pred))
    
    # 2. Matriz de Confusão
    conf_matrix = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    
    print("\n2. Explicação da Matriz de Confusão:")
    print(f"- Verdadeiros Negativos (pagamentos previstos corretamente): {tn}")
    print(f"- Falsos Positivos (alarmes falsos de default): {fp}")
    print(f"- Falsos Negativos (defaults não detectados): {fn}")
    print(f"- Verdadeiros Positivos (defaults previstos corretamente): {tp}")
    
    # 3. Métricas de Negócio
    print("\n3. Métricas de Negócio:")
    print(f"- Taxa de Detecção de Default: {tp/(tp+fn):.1%}")
    print(f"- Taxa de Falso Alarme: {fp/(fp+tn):.1%}")
    print(f"- Precisão na Detecção de Default: {tp/(tp+fp):.1%}")
    
    # 4. AUC-ROC
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    print(f"\n4. AUC-ROC Score: {auc_roc:.3f}")
    
    return conf_matrix, auc_roc

# 1. Carregar os dados preparados
print("\n1. Carregando dados preparados...")
model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
X_train = np.load(os.path.join(model_dir, 'X_train_transformed.npy'))
X_test = np.load(os.path.join(model_dir, 'X_test_transformed.npy'))
y_train = np.load(os.path.join(model_dir, 'y_train.npy'))
y_test = np.load(os.path.join(model_dir, 'y_test.npy'))

with open(os.path.join(model_dir, 'feature_names.pkl'), 'rb') as f:
    feature_names = pickle.load(f)

# 2. Aplicar SMOTE para balancear os dados
print("\n2. Balanceando os dados com SMOTE...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("\nDistribuição das classes após SMOTE:")
print(f"Não Default: {sum(y_train_balanced == 0)}")
print(f"Default: {sum(y_train_balanced == 1)}")

# 3. Treinar e avaliar cada modelo
resultados = {}

# 3.1 Regressão Logística
print("\n3.1 Treinando Regressão Logística...")
print("Este é o modelo mais simples e interpretável:")
print("- Tenta encontrar uma linha que separe os defaults dos não-defaults")
print("- Bom para entender quais variáveis são mais importantes")
print("- Rápido para treinar e fácil de interpretar")

log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_balanced, y_train_balanced)
y_pred_log = log_reg.predict(X_test)
y_pred_proba_log = log_reg.predict_proba(X_test)[:, 1]

conf_matrix_log, auc_log = avaliar_modelo(y_test, y_pred_log, y_pred_proba_log, "Regressão Logística")
resultados['Regressão Logística'] = {'conf_matrix': conf_matrix_log, 'auc': auc_log}

# 3.2 Random Forest
print("\n3.2 Treinando Random Forest...")
print("Este modelo é um conjunto de árvores de decisão:")
print("- Cada árvore 'vota' se acha que será default ou não")
print("- Bom para capturar relações não lineares")
print("- Mais robusto que a Regressão Logística")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_balanced, y_train_balanced)
y_pred_rf = rf.predict(X_test)
y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]

conf_matrix_rf, auc_rf = avaliar_modelo(y_test, y_pred_rf, y_pred_proba_rf, "Random Forest")
resultados['Random Forest'] = {'conf_matrix': conf_matrix_rf, 'auc': auc_rf}

# 3.3 XGBoost
print("\n3.3 Treinando XGBoost...")
print("Este é o modelo mais avançado:")
print("- Aprende gradualmente com seus erros")
print("- Geralmente tem a melhor performance")
print("- Mais complexo de ajustar")

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    random_state=42,
    n_estimators=100
)
xgb_model.fit(X_train_balanced, y_train_balanced)
y_pred_xgb = xgb_model.predict(X_test)
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

conf_matrix_xgb, auc_xgb = avaliar_modelo(y_test, y_pred_xgb, y_pred_proba_xgb, "XGBoost")
resultados['XGBoost'] = {'conf_matrix': conf_matrix_xgb, 'auc': auc_xgb}

# 4. Comparação dos modelos
print("\n4. Comparação Final dos Modelos:")
print("\nAUC-ROC Scores:")
for modelo, res in resultados.items():
    print(f"{modelo}: {res['auc']:.3f}")

# 5. Visualizações
print("\n5. Criando visualizações comparativas...")

# 5.1 Matriz de Confusão para cada modelo
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (modelo, res) in enumerate(resultados.items()):
    sns.heatmap(res['conf_matrix'], annot=True, fmt='d', cmap='Blues', ax=axes[i])
    axes[i].set_title(f'Matriz de Confusão\n{modelo}')
    axes[i].set_ylabel('Real')
    axes[i].set_xlabel('Previsto')

plt.tight_layout()
plt.savefig(os.path.join(model_dir, 'comparison_confusion_matrices.png'))
plt.close()

# 5.2 Importância das Features para cada modelo
def plot_feature_importance(model, model_name):
    if model_name == "Regressão Logística":
        importance = abs(model.coef_[0])
    elif model_name == "Random Forest":
        importance = model.feature_importances_
    else:  # XGBoost
        importance = model.feature_importances_
    
    feature_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_imp.head(10))
    plt.title(f'Top 10 Features Mais Importantes - {model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, f'feature_importance_{model_name}.png'))
    plt.close()

plot_feature_importance(log_reg, "Regressão Logística")
plot_feature_importance(rf, "Random Forest")
plot_feature_importance(xgb_model, "XGBoost")

# 6. Salvar os modelos
print("\n6. Salvando os modelos...")
models = {
    'logistic': log_reg,
    'random_forest': rf,
    'xgboost': xgb_model
}

for name, model in models.items():
    with open(os.path.join(model_dir, f'{name}_model.pkl'), 'wb') as f:
        pickle.dump(model, f)

print("\nAnálise completa! Todos os modelos foram salvos na pasta 'models'")
print("Verifique as visualizações geradas para uma comparação detalhada dos modelos.")
