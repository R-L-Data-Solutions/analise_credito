import pandas as pd
import numpy as np
import requests
import zipfile
import os
from tqdm import tqdm

class LendingClubScraper:
    """
    Classe para baixar e preparar dados do Lending Club para análise de crédito.
    """
    def __init__(self):
        # Definir caminhos relativos ao diretório do projeto
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.project_dir, 'data', 'raw')
        self.processed_dir = os.path.join(self.project_dir, 'data', 'processed')
        
        self.raw_file = 'LoanStats_2018Q4.csv'
        self.processed_file = 'lending_club_processed.csv'
        
        # Criar diretórios se não existirem
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def download_data(self):
        """
        Baixa os dados do Lending Club.
        Usaremos o dataset de 2018 Q4 que está publicamente disponível.
        """
        url = "https://resources.lendingclub.com/LoanStats_2018Q4.csv.zip"
        zip_path = os.path.join(self.data_dir, 'loan_stats.zip')
        
        print("Baixando dados do Lending Club...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as file, tqdm(
            desc="Download Progress",
            total=total_size,
            unit='iB',
            unit_scale=True
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)
        
        print("Extraindo arquivo...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
        
        # Remover arquivo zip após extração
        os.remove(zip_path)

    def process_data(self):
        """
        Processa os dados brutos do Lending Club para análise.
        Seleciona features relevantes e prepara target variable.
        """
        print("Processando dados...")
        
        # Carregar dados (pulando linhas de cabeçalho desnecessárias)
        df = pd.read_csv(os.path.join(self.data_dir, self.raw_file), skiprows=1)
        
        # Selecionar features relevantes
        features = [
            'loan_amnt',           # Valor do empréstimo
            'term',                # Prazo
            'int_rate',           # Taxa de juros
            'grade',              # Grade de crédito atribuído
            'emp_length',         # Tempo de emprego
            'home_ownership',     # Tipo de residência
            'annual_inc',         # Renda anual
            'purpose',            # Finalidade do empréstimo
            'addr_state',         # Estado
            'dti',               # Debt-to-Income ratio
            'delinq_2yrs',       # Número de delinquências nos últimos 2 anos
            'earliest_cr_line',   # Data da primeira linha de crédito
            'inq_last_6mths',    # Consultas nos últimos 6 meses
            'open_acc',          # Número de contas abertas
            'pub_rec',           # Registros públicos
            'revol_bal',         # Saldo rotativo
            'revol_util',        # Utilização do crédito rotativo
            'total_acc',         # Total de contas
            'loan_status'        # Status do empréstimo (target)
        ]
        
        df_processed = df[features].copy()
        
        # Criar target variable (1 = default, 0 = não default)
        df_processed['default'] = df_processed['loan_status'].map(
            lambda x: 1 if x in ['Charged Off', 'Default', 'Late (31-120 days)', 
                               'Late (16-30 days)'] else 0
        )
        
        # Remover a coluna loan_status original
        df_processed.drop('loan_status', axis=1, inplace=True)
        
        # Limpar e converter tipos de dados
        df_processed['int_rate'] = df_processed['int_rate'].str.rstrip('%').astype(float)
        df_processed['revol_util'] = df_processed['revol_util'].str.rstrip('%').astype(float)
        df_processed['emp_length'] = df_processed['emp_length'].str.extract('(\d+)').fillna(0).astype(int)
        
        # Salvar dados processados
        output_path = os.path.join(self.processed_dir, self.processed_file)
        df_processed.to_csv(output_path, index=False)
        
        print(f"\nDados processados e salvos em: {output_path}")
        print(f"Dimensões do dataset: {df_processed.shape}")
        return df_processed

    def get_data(self):
        """
        Função principal para obter os dados processados.
        """
        try:
            self.download_data()
            df = self.process_data()
            
            print("\nInformações do dataset:")
            print(df.info())
            
            print("\nPrimeiras linhas do dataset:")
            print(df.head())
            
            print("\nEstatísticas básicas:")
            print(df.describe())
            
            # Taxa de default
            default_rate = (df['default'].mean() * 100)
            print(f"\nTaxa de default no dataset: {default_rate:.2f}%")
            
            return df
            
        except Exception as e:
            print(f"Erro ao obter os dados: {e}")
            return None

if __name__ == "__main__":
    scraper = LendingClubScraper()
    df = scraper.get_data()
