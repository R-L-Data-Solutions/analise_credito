# Estrutura do Projeto

## Organização das Pastas

### 📱 interface/
Interface do usuário
- `credit_analysis_app.py`: Interface principal em Streamlit
- `analisar_cliente.py`: Interface de análise de cliente

### 🔄 core/
Núcleo do sistema
- `enhanced_model.py`: Modelo principal de análise
- `train_logistic_model.py`: Treinamento do modelo

### 📊 analysis/
Módulos de análise
- `analysis_interest_rate.py`: Análise de taxas de juros
- `feature_analysis.py`: Análise de características
- `exploratory_analysis.py`: Análise exploratória

### 🔧 data_processing/
Processamento de dados
- `lending_club_scraper.py`: Coleta de dados
- `prepare_model_data.py`: Preparação dos dados

### 🛠️ utils/
Funções utilitárias e helpers

## Como Executar

1. Interface Principal:
```bash
streamlit run src/interface/credit_analysis_app.py
```

2. Análise de Cliente:
```bash
python src/interface/analisar_cliente.py
```

3. Treinar Modelo:
```bash
python src/core/train_logistic_model.py
```
