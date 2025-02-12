# Análise de Risco de Crédito - Principais Descobertas

## 1. Perfil Geral de Risco
- Taxa média de default: 13.45%
- De cada 100 empréstimos, aproximadamente 13 não são pagos
- Isso representa um risco significativo que precisa ser gerenciado
- Total da carteira analisada: mais de 128 mil empréstimos

## 2. Fatores de Risco Identificados

### Tipo de Moradia
- Pessoas com casa própria têm menor risco de default (~12%)
- Inquilinos apresentam risco ligeiramente maior (~14%)
- O tipo de moradia é um bom indicador de estabilidade financeira
- A maioria dos empréstimos é para pessoas com imóvel financiado

### Renda
- Clientes com renda mais alta têm menor probabilidade de default
- Existe uma clara relação entre renda e capacidade de pagamento
- A renda média dos bons pagadores é significativamente maior
- Podemos usar isso para ajustar os limites de crédito

### Taxa de Juros
- Empréstimos com taxas mais altas têm maior probabilidade de default
- As taxas variam de 6% a 31% ao ano
- Existe um "efeito bola de neve": maior risco → taxa mais alta → maior chance de default
- Importante encontrar um equilíbrio na precificação

### Tempo de Emprego
- O tempo de emprego tem relação com o risco de default
- Pessoas com maior estabilidade profissional tendem a ser melhores pagadores
- Este é um bom indicador para avaliação de crédito

## 3. Recomendações para Política de Crédito

### Imediatas
1. Considerar o tipo de moradia na avaliação de risco
2. Estabelecer limites de empréstimo baseados na renda
3. Revisar a política de taxas de juros para alto risco
4. Dar peso maior para tempo de emprego na análise

### Médio Prazo
1. Desenvolver sistema de pontuação (scoring) baseado nos fatores identificados
2. Criar faixas de risco com políticas específicas
3. Implementar alertas para combinações de alto risco

## 4. Próximos Passos

### Desenvolvimento do Modelo
1. Preparar os dados para modelagem
   - Tratar valores faltantes
   - Codificar variáveis categóricas
   - Normalizar variáveis numéricas

2. Criar modelo de classificação
   - Testar diferentes algoritmos (Random Forest, XGBoost, etc.)
   - Validar resultados com cross-validation
   - Otimizar hiperparâmetros

3. Avaliar performance
   - Métricas de classificação (AUC-ROC, precisão, recall)
   - Análise de importância das features
   - Validação com dados de teste

### Implementação
1. Criar API para o modelo
2. Desenvolver interface para usuários
3. Estabelecer monitoramento de performance

## 5. Conclusões
- Os dados mostram padrões claros de risco
- Existem várias features úteis para predição
- Um modelo de ML pode automatizar e melhorar o processo
- O monitoramento contínuo será essencial

## 6. Anexos
- Todos os gráficos estão disponíveis na pasta `/plots`
- Códigos de análise na pasta `/src`
- Dados processados em `/data/processed`
