from enhanced_model import treinar_modelo_avancado, analisar_novo_caso

def formatar_moeda(valor):
    """Formata valor em reais"""
    return f"R$ {valor:,.2f}"

def analisar_cliente():
    """Interface simples para análise de crédito"""
    print("\n=== SISTEMA DE ANÁLISE DE CRÉDITO ===\n")
    
    # 1. Dados do Empréstimo
    print("--- DADOS DO EMPRÉSTIMO ---")
    loan_amnt = float(input("Valor do empréstimo desejado: R$ ").replace(',', '.'))
    int_rate = float(input("Taxa de juros (%): ").replace(',', '.'))
    annual_inc = float(input("Renda anual: R$ ").replace(',', '.'))
    
    # 2. Histórico de Pagamentos
    print("\n--- HISTÓRICO DE PAGAMENTOS ---")
    dias_atraso = float(input("Média de dias de atraso em pagamentos anteriores: ").replace(',', '.'))
    parcelas_pontuais = float(input("Porcentagem de parcelas pagas em dia (0-100): ").replace(',', '.')) / 100
    maior_atraso = float(input("Maior atraso registrado (em dias): ").replace(',', '.'))
    
    # 3. Relacionamento com o Banco
    print("\n--- RELACIONAMENTO COM O BANCO ---")
    tempo_conta = float(input("Tempo de conta (anos): ").replace(',', '.'))
    saldo_medio = float(input("Saldo médio nos últimos 3 meses: R$ ").replace(',', '.'))
    usa_cheque = input("Usa cheque especial? (S/N): ").upper() == 'S'
    
    # 4. Dados de Crédito
    print("\n--- DADOS DE CRÉDITO ---")
    score = float(input("Score de crédito (0-1000): ").replace(',', '.'))
    consultas = int(input("Número de consultas ao CPF nos últimos 6 meses: "))
    outros_emprestimos = input("Possui outros empréstimos? (S/N): ").upper() == 'S'
    
    # Organizar dados
    dados_cliente = {
        'loan_amnt': loan_amnt,
        'int_rate': int_rate,
        'annual_inc': annual_inc,
        'dias_atraso_media': dias_atraso,
        'parcelas_pagas_pontualmente': parcelas_pontuais,
        'maior_atraso': maior_atraso,
        'tempo_conta_anos': tempo_conta,
        'saldo_medio': saldo_medio,
        'usa_cheque_especial': 1 if usa_cheque else 0,
        'score_credito': score,
        'consultas_cpf_6m': consultas,
        'tem_outros_emprestimos': 1 if outros_emprestimos else 0
    }
    
    # Treinar modelo e fazer análise
    print("\nAnalisando dados...")
    model, feature_names = treinar_modelo_avancado()
    resultado = analisar_novo_caso(model, feature_names, dados_cliente)
    
    # Mostrar resultado
    print("\n=== RESULTADO DA ANÁLISE ===")
    print(f"\nPrevisão: {resultado['previsao']}")
    print(f"Probabilidade de Default: {resultado['probabilidade_default']:.1f}%")
    
    if resultado['fatores_risco']:
        print("\nFatores de Risco Identificados:")
        for fator in resultado['fatores_risco']:
            print(f"- {fator}")
    else:
        print("\nNenhum fator de risco significativo identificado!")
    
    # Análise adicional
    print("\n=== ANÁLISE FINANCEIRA ===")
    renda_mensal = annual_inc / 12
    print(f"Renda Mensal: {formatar_moeda(renda_mensal)}")
    
    # Calcular comprometimento de renda
    parcela_estimada = loan_amnt * (1 + int_rate/100) / 12  # Estimativa simples
    comprometimento = (parcela_estimada / renda_mensal) * 100
    print(f"Parcela Estimada: {formatar_moeda(parcela_estimada)}")
    print(f"Comprometimento de Renda: {comprometimento:.1f}%")
    
    if comprometimento > 30:
        print("\nALERTA: Comprometimento de renda acima do recomendado (30%)!")
    
    # Recomendação final
    print("\n=== RECOMENDAÇÃO ===")
    if resultado['previsao'] == 'BAIXO RISCO' and comprometimento <= 30:
        print("✅ APROVADO: Empréstimo pode ser concedido")
    elif resultado['previsao'] == 'BAIXO RISCO':
        print("⚠️ APROVADO COM RESSALVAS: Comprometimento de renda alto")
    else:
        print("❌ NÃO RECOMENDADO: Alto risco de default")

if __name__ == "__main__":
    try:
        analisar_cliente()
    except ValueError as e:
        print("\nErro: Por favor, insira apenas números válidos.")
    except Exception as e:
        print(f"\nErro inesperado: {str(e)}")
