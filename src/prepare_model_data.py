import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import pickle
import os

# Criar diretório para salvar os objetos do modelo
print("0. Criando diretório para os objetos do modelo...")
model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 1. Carregar e preparar os dados
print("\n1. Carregando os dados...")
df = pd.read_csv(r'C:\Users\ronaldo.pereira\Desktop\Projetos\credit_analysis\data\processed\lending_club_processed.csv')

# 2. Selecionar features relevantes
print("\n2. Selecionando features relevantes...")
numeric_features = [
    'loan_amnt',      # Valor do empréstimo
    'int_rate',       # Taxa de juros
    'annual_inc',     # Renda anual
    'dti',           # Relação dívida/renda
    'inq_last_6mths', # Consultas nos últimos 6 meses
    'emp_length',     # Tempo de emprego
    'revol_util'      # Utilização de crédito rotativo
]

categorical_features = [
    'grade',          # Nota de crédito
    'home_ownership', # Tipo de moradia
    'purpose',        # Finalidade do empréstimo
    'term'           # Prazo
]

# 3. Criar features derivadas
print("\n3. Criando features derivadas...")
df['loan_to_income'] = (df['loan_amnt'] / df['annual_inc']).clip(upper=1)  # Limitando a 100% da renda
df['payment_to_income'] = ((df['loan_amnt'] * (df['int_rate']/100)) / df['annual_inc']).clip(upper=1)
numeric_features.extend(['loan_to_income', 'payment_to_income'])

# 4. Tratar valores faltantes
print("\n4. Tratando valores faltantes...")
# Para features numéricas: preencher com a mediana
for col in numeric_features:
    df[col] = df[col].fillna(df[col].median())
    print(f"- Valores faltantes em {col}: {df[col].isnull().sum()}")

# Para features categóricas: preencher com a moda (valor mais frequente)
for col in categorical_features:
    df[col] = df[col].fillna(df[col].mode()[0])
    print(f"- Valores faltantes em {col}: {df[col].isnull().sum()}")

# 5. Separar features (X) e target (y)
print("\n5. Separando features e target...")
X = df[numeric_features + categorical_features]
y = df['default']

# 6. Dividir em treino e teste
print("\n6. Dividindo em treino e teste...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"- Tamanho do conjunto de treino: {X_train.shape}")
print(f"- Tamanho do conjunto de teste: {X_test.shape}")

# 7. Criar pipeline de transformação
print("\n7. Criando pipeline de transformação...")
numeric_transformer = StandardScaler()  # Padroniza os dados numéricos
categorical_transformer = OneHotEncoder(drop='first')  # Converte categorias em colunas binárias

# Combinar transformadores
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 8. Ajustar e transformar os dados de treino
print("\n8. Transformando os dados...")
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Obter nomes das features após transformação
cat_feature_names = []
for i, feature in enumerate(categorical_features):
    # Obter categorias únicas para cada feature categórica
    unique_cats = X_train[feature].unique()
    # Adicionar nomes das features (excluindo primeira categoria devido a drop='first')
    cat_feature_names.extend([f"{feature}_{cat}" for cat in unique_cats[1:]])

feature_names = numeric_features + cat_feature_names

# 9. Salvar objetos importantes
print("\n9. Salvando objetos para uso futuro...")
with open(os.path.join(model_dir, 'preprocessor.pkl'), 'wb') as f:
    pickle.dump(preprocessor, f)

with open(os.path.join(model_dir, 'feature_names.pkl'), 'wb') as f:
    pickle.dump(feature_names, f)

# Salvar também os dados transformados
np.save(os.path.join(model_dir, 'X_train_transformed.npy'), X_train_transformed)
np.save(os.path.join(model_dir, 'X_test_transformed.npy'), X_test_transformed)
np.save(os.path.join(model_dir, 'y_train.npy'), y_train.values)
np.save(os.path.join(model_dir, 'y_test.npy'), y_test.values)

print("\nPreparação dos dados concluída! Os dados estão prontos para o modelo.")
print(f"Dimensões finais dos dados de treino: {X_train_transformed.shape}")

# 10. Mostrar algumas estatísticas dos dados transformados
print("\n10. Estatísticas dos dados transformados:")
print("\nFeatures numéricas (após padronização):")
for i, feature in enumerate(numeric_features):
    mean = X_train_transformed[:, i].mean()
    std = X_train_transformed[:, i].std()
    print(f"{feature}:")
    print(f"  - Média: {mean:.3f}")
    print(f"  - Desvio Padrão: {std:.3f}")
