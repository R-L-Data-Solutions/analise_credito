import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class CreditAnalysis:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_path = os.path.join(self.project_dir, 'data', 'processed', 'lending_club_processed.csv')
        self.plots_dir = os.path.join(self.project_dir, 'plots')
        
        # Criar diretório para plots se não existir
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)
        
        # Configurações de visualização
        plt.style.use('seaborn')
        sns.set_palette('husl')

    def load_data(self):
        """Carrega os dados processados"""
        print("Carregando dados...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dimensões do dataset: {self.df.shape}")
        print("\nPrimeiras linhas:")
        print(self.df.head())
        return self.df

    def analyze_default_rate(self):
        """Análise da taxa de default geral"""
        print("\n=== Análise da Taxa de Default ===")
        
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.df, x='default')
        plt.title('Distribuição de Default')
        plt.xlabel('Default (0=Não, 1=Sim)')
        plt.ylabel('Número de Empréstimos')
        plt.savefig(os.path.join(self.plots_dir, '1_default_distribution.png'))
        plt.close()

        default_rate = (self.df['default'].mean() * 100)
        print(f"Taxa de default total: {default_rate:.2f}%")

    def analyze_credit_grades(self):
        """Análise por grade de crédito"""
        print("\n=== Análise por Grade de Crédito ===")
        
        default_by_grade = self.df.groupby('grade')['default'].agg(['mean', 'count']).sort_index()
        
        plt.figure(figsize=(12, 6))
        default_by_grade['mean'].plot(kind='bar')
        plt.title('Taxa de Default por Grade de Crédito')
        plt.xlabel('Grade')
        plt.ylabel('Taxa de Default')
        plt.savefig(os.path.join(self.plots_dir, '2_default_by_grade.png'))
        plt.close()

        print("\nTaxa de default por grade:")
        print(default_by_grade['mean'].multiply(100).round(2))
        print("\nNúmero de empréstimos por grade:")
        print(default_by_grade['count'])

    def analyze_loan_amount_and_interest(self):
        """Análise de valor do empréstimo e taxa de juros"""
        print("\n=== Análise de Valor e Taxa de Juros ===")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Boxplot do valor do empréstimo por default
        sns.boxplot(data=self.df, x='default', y='loan_amnt', ax=ax1)
        ax1.set_title('Valor do Empréstimo vs Default')
        ax1.set_xlabel('Default (0=Não, 1=Sim)')
        ax1.set_ylabel('Valor do Empréstimo ($)')

        # Boxplot da taxa de juros por default
        sns.boxplot(data=self.df, x='default', y='int_rate', ax=ax2)
        ax2.set_title('Taxa de Juros vs Default')
        ax2.set_xlabel('Default (0=Não, 1=Sim)')
        ax2.set_ylabel('Taxa de Juros (%)')

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, '3_loan_amount_interest.png'))
        plt.close()

        # Estatísticas
        stats = self.df.groupby('default')[['loan_amnt', 'int_rate']].agg(['mean', 'median'])
        print("\nEstatísticas por status de default:")
        print(stats.round(2))

    def analyze_correlations(self):
        """Análise de correlações"""
        print("\n=== Análise de Correlações ===")
        
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        correlation_matrix = self.df[numeric_cols].corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Matriz de Correlação - Features Numéricas')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, '4_correlation_matrix.png'))
        plt.close()

        # Mostrar correlações mais fortes com default
        correlations_with_default = correlation_matrix['default'].sort_values(ascending=False)
        print("\nCorrelações mais fortes com default:")
        print(correlations_with_default)

    def analyze_loan_purpose(self):
        """Análise por finalidade do empréstimo"""
        print("\n=== Análise por Finalidade do Empréstimo ===")
        
        default_by_purpose = self.df.groupby('purpose')['default'].agg(['mean', 'count'])
        default_by_purpose = default_by_purpose.sort_values('mean', ascending=False)

        plt.figure(figsize=(15, 6))
        sns.barplot(data=default_by_purpose.reset_index(), x='purpose', y='mean')
        plt.title('Taxa de Default por Finalidade do Empréstimo')
        plt.xticks(rotation=45)
        plt.xlabel('Finalidade')
        plt.ylabel('Taxa de Default')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, '5_default_by_purpose.png'))
        plt.close()

        print("\nTaxa de default por finalidade:")
        print(default_by_purpose['mean'].multiply(100).round(2))
        print("\nNúmero de empréstimos por finalidade:")
        print(default_by_purpose['count'])

    def run_all_analyses(self):
        """Executa todas as análises"""
        self.load_data()
        self.analyze_default_rate()
        self.analyze_credit_grades()
        self.analyze_loan_amount_and_interest()
        self.analyze_correlations()
        self.analyze_loan_purpose()
        print(f"\nTodos os gráficos foram salvos em: {self.plots_dir}")

if __name__ == "__main__":
    analyzer = CreditAnalysis()
    analyzer.run_all_analyses()
