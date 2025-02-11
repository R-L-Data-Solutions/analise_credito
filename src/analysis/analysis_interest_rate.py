import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configurações de estilo
plt.style.use('seaborn')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Carregar os dados
df = pd.read_csv(r'C:\Users\ronaldo.pereira\Desktop\Projetos\credit_analysis\data\processed\lending_club_processed.csv')

# Dicionário de tradução para tipos de moradia
traducao_moradia = {
    'RENT': 'Alugada',
    'MORTGAGE': 'Financiada',
    'OWN': 'Própria',
    'OTHER': 'Outros',
    'NONE': 'Não Informado',
    'ANY': 'Qualquer'
}

# Criar figura com três subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# 1. Distribuição de default por tempo de emprego
default_by_emp = df.groupby('emp_length')['default'].mean() * 100
default_by_emp.plot(kind='bar', ax=ax1, color='skyblue')
ax1.set_title('Taxa de Default por\nTempo de Emprego')
ax1.set_xlabel('Anos de Emprego')
ax1.set_ylabel('Taxa de Default (%)')
ax1.tick_params(axis='x', rotation=45)

# Adicionar valores nas barras
for i, v in enumerate(default_by_emp):
    ax1.text(i, v + 0.5, f'{v:.1f}%', ha='center')

# 2. Boxplot de renda por status de default
sns.boxplot(data=df, x='default', y='annual_inc', ax=ax2,
            palette=['lightgreen', 'lightcoral'])
ax2.set_title('Distribuição de Renda por\nStatus do Empréstimo')
ax2.set_xlabel('Status do Empréstimo')
ax2.set_ylabel('Renda Anual (USD)')
ax2.set_xticklabels(['Pago', 'Não Pago'])

# Formatar eixo y para mostrar milhares
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# 3. Taxa de default por tipo de moradia
# Traduzir os tipos de moradia
df['home_ownership_pt'] = df['home_ownership'].map(traducao_moradia)
default_by_home = df.groupby('home_ownership_pt')['default'].agg(['mean', 'count'])
default_by_home['mean'] = default_by_home['mean'] * 100

# Criar barras com duas informações
ax3.bar(default_by_home.index, default_by_home['mean'], color='lightblue')
ax3.set_title('Taxa de Default por\nTipo de Moradia')
ax3.set_xlabel('Tipo de Moradia')
ax3.set_ylabel('Taxa de Default (%)')
ax3.tick_params(axis='x', rotation=45)  # Rotacionar labels para melhor legibilidade

# Adicionar valores nas barras
for i, (v, c) in enumerate(zip(default_by_home['mean'], default_by_home['count'])):
    ax3.text(i, v + 0.5, f'{v:.1f}%\n(n={c:,})', ha='center')

plt.suptitle('Análise de Default por Perfil do Cliente', fontsize=16, y=1.05)

# Adicionar texto explicativo
fig.text(0.02, 0.02, 
         'Observações:\n' + 
         '- O gráfico da esquerda mostra como a taxa de default varia com o tempo de emprego\n' + 
         '- O gráfico do meio compara a distribuição de renda entre empréstimos pagos e não pagos\n' + 
         '- O gráfico da direita mostra a taxa de default por tipo de moradia', 
         fontsize=10, ha='left')

plt.tight_layout()
plt.savefig(r'C:\Users\ronaldo.pereira\Desktop\Projetos\credit_analysis\plots/client_profile_analysis.png',
            bbox_inches='tight', dpi=300)
plt.show()

# Imprimir algumas estatísticas
print("\nEstatísticas por tempo de emprego:")
estatisticas = df.groupby('emp_length').agg({
    'default': ['mean', 'count'],
    'annual_inc': 'mean',
    'loan_amnt': 'mean'
}).round(3)

estatisticas.columns = ['Taxa de Default', 'Quantidade', 'Renda Média (USD)', 'Valor Médio Empréstimo (USD)']
print(estatisticas)

# Vamos ver os números exatos
total_emprestimos = len(df)
emprestimos_default = df['default'].sum()
taxa_default_geral = (emprestimos_default / total_emprestimos) * 100

print(f"Total de empréstimos: {total_emprestimos:,}")
print(f"Empréstimos em default: {emprestimos_default:,}")
print(f"Taxa de default geral: {taxa_default_geral:.2f}%")

# Vamos ver também por tipo de moradia para ficar mais claro
analise_detalhada = df.groupby('home_ownership_pt').agg({
    'default': ['count', 'sum', lambda x: (x.sum()/len(x))*100]
}).round(2)

analise_detalhada.columns = ['Total Empréstimos', 'Qtd em Default', 'Taxa Default (%)']
print("\nAnálise detalhada por tipo de moradia:")
print(analise_detalhada)