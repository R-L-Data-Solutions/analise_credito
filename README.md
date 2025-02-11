# Sistema de Análise de Crédito

Sistema profissional para análise de risco de crédito com interface gráfica em Streamlit, incluindo simulação de empréstimos e análise de risco.

## 🚀 Funcionalidades

### Análise de Crédito
- Avaliação de risco baseada em múltiplos fatores
- Score de crédito personalizado
- Detecção automática de fatores de risco
- Histórico de pagamentos e relacionamento bancário

### Simulação de Empréstimo
- Cálculo de parcelas com juros compostos
- Taxa de juros dinâmica baseada no perfil do cliente
- Simulação detalhada mês a mês
- Análise de comprometimento de renda

### Interface Profissional
- Design limpo e intuitivo em Streamlit
- Visualização clara dos resultados
- Métricas importantes em destaque
- Tabela de evolução do empréstimo

## 📊 Modelo de Análise

O sistema considera diversos fatores:

### Dados Financeiros
- Valor do empréstimo solicitado
- Renda mensal e anual
- Comprometimento de renda
- Saldo médio em conta

### Histórico do Cliente
- Score de crédito (0-1000)
- Histórico de pagamentos
- Tempo de relacionamento com o banco
- Consultas recentes ao CPF

### Análise de Risco
- Uso de cheque especial
- Outros empréstimos ativos
- Atrasos em pagamentos anteriores
- Perfil de crédito

## 🛠️ Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/credit-analysis.git
cd credit-analysis
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Execute a aplicação:
```bash
streamlit run src/interface/credit_analysis_app.py
```

## 📁 Estrutura do Projeto

```
credit-analysis/
├── src/
│   ├── interface/          # Interface do usuário
│   │   ├── credit_analysis_app.py
│   │   └── analisar_cliente.py
│   │
│   ├── core/              # Núcleo do sistema
│   │   ├── enhanced_model.py
│   │   └── train_logistic_model.py
│   │
│   ├── analysis/          # Módulos de análise
│   │   ├── analysis_interest_rate.py
│   │   ├── feature_analysis.py
│   │   └── exploratory_analysis.py
│   │
│   ├── data_processing/   # Processamento de dados
│   │   ├── lending_club_scraper.py
│   │   └── prepare_model_data.py
│   │
│   └── utils/            # Funções utilitárias
│
├── data/                 # Dados para treinamento
├── models/              # Modelos treinados
└── requirements.txt     # Dependências
```

## ⚙️ Configuração do Modelo

### Taxa de Juros
- **Taxa Base**: 19.56% a.a.

### Ajuste por Prazo
- Até 12 meses: +0%
- 13-24 meses: +2%
- 25-36 meses: +4%
- Acima: +6%

### Ajuste por Score
- 800+: -4%
- 700-799: -2%
- 600-699: +0%
- 500-599: +2%
- Abaixo: +4%

## 📝 Como Usar

1. **Dados do Cliente**
   - Preencha informações financeiras
   - Insira histórico de pagamentos
   - Informe relacionamento bancário

2. **Simulação**
   - Escolha valor do empréstimo
   - Selecione prazo desejado
   - Veja simulação detalhada

3. **Análise**
   - Verifique resultado da análise
   - Consulte fatores de risco
   - Avalie comprometimento de renda

## 🤝 Contribuindo

1. Fork o projeto
2. Crie sua Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a Branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## ✨ Melhorias Futuras

- [ ] Integração com APIs de bureaus de crédito
- [ ] Análise de dados bancários em tempo real
- [ ] Suporte a diferentes tipos de empréstimo
- [ ] Dashboard administrativo
- [ ] Relatórios em PDF
- [ ] Histórico de análises realizadas
- [ ] Integração com sistemas bancários
