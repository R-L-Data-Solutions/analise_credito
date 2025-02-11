import pandas as pd
import numpy as np

# Carregar dados
df = pd.read_csv(r'C:\Users\ronaldo.pereira\Desktop\Projetos\credit_analysis\data\processed\lending_club_processed.csv')

print("=== Análise de Features para Modelo de ML ===\n")

# Calcular correlações com default (apenas para variáveis numéricas)
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlacoes = df[numeric_cols].corr()['default'].sort_values(ascending=False)
print("\nCorrelações com Default (ordenadas):")
print(correlacoes.round(3))

# Análise das variáveis categóricas
print("\nAnálise de Variáveis Categóricas:")

categorical_cols = ['grade', 'home_ownership', 'purpose', 'term']
for col in categorical_cols:
    print(f"\nTaxa de Default por {col}:")
    default_rate = df.groupby(col)['default'].agg(['count', 'mean']).round(3)
    default_rate['mean'] = default_rate['mean'] * 100  # Converter para percentagem
    default_rate.columns = ['Quantidade', 'Taxa de Default (%)']
    print(default_rate)

# Criar e analisar features derivadas
print("\nFeatures Derivadas:")
df['loan_to_income'] = df['loan_amnt'] / df['annual_inc']
df['payment_to_income'] = (df['loan_amnt'] * (df['int_rate']/100)) / df['annual_inc']

print("\nCorrelações das novas features com default:")
new_features = ['loan_to_income', 'payment_to_income']
for feat in new_features:
    corr = df[feat].corr(df['default'])
    print(f"- {feat}: {corr:.3f}")