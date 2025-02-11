from predict_new_loan import analisar_emprestimo, imprimir_resultado

def calcular_metricas(valor_emprestimo, renda_anual, divida_atual):
    """
    Calcula as métricas necessárias para a análise.
    
    Parâmetros:
    - valor_emprestimo: Quanto a pessoa quer emprestar
    - renda_anual: Renda anual da pessoa
    - divida_atual: Valor total das dívidas atuais
    """
    # Calcular DTI (Debt-to-Income)
    dti = (divida_atual / renda_anual) * 100
    
    # Calcular Loan-to-Income
    loan_to_income = valor_emprestimo / renda_anual
    
    return dti, loan_to_income

def analisar_novo_emprestimo():
    """Exemplo de como analisar um novo empréstimo."""
    print("=== ANÁLISE DE NOVO EMPRÉSTIMO ===")
    
    # 1. Coletar dados básicos
    print("\n1. DADOS DO EMPRÉSTIMO:")
    valor_emprestimo = 15000
    taxa_juros = 12.5
    renda_anual = 60000
    divida_atual = 10000
    grade_credito = 'B'
    finalidade = 'debt_consolidation'
    
    print(f"Valor do Empréstimo: R$ {valor_emprestimo:,.2f}")
    print(f"Taxa de Juros: {taxa_juros}%")
    print(f"Renda Anual: R$ {renda_anual:,.2f}")
    print(f"Dívida Atual: R$ {divida_atual:,.2f}")
    print(f"Grade de Crédito: {grade_credito}")
    print(f"Finalidade: {finalidade}")
    
    # 2. Calcular métricas
    print("\n2. MÉTRICAS CALCULADAS:")
    dti, loan_to_income = calcular_metricas(valor_emprestimo, renda_anual, divida_atual)
    print(f"DTI (Debt-to-Income): {dti:.1f}%")
    print(f"Loan-to-Income: {loan_to_income:.3f}")
    
    # 3. Preparar dados para o modelo
    dados_emprestimo = {
        'loan_amnt': valor_emprestimo,
        'int_rate': taxa_juros,
        'annual_inc': renda_anual,
        'dti': dti,
        'loan_to_income': loan_to_income,
        'grade': grade_credito,
        'purpose': finalidade
    }
    
    # 4. Fazer a análise
    print("\n3. ANÁLISE DE RISCO:")
    resultado = analisar_emprestimo(dados_emprestimo)
    imprimir_resultado(resultado)
    
    # 5. Explicar o resultado
    print("\n4. EXPLICAÇÃO:")
    if resultado['previsao'] == 'ALTO RISCO':
        print("Este empréstimo foi classificado como ALTO RISCO porque:")
        if dti > 20:
            print(f"- DTI está alto ({dti:.1f}% > 20%)")
        if loan_to_income > 0.3:
            print(f"- Empréstimo é alto em relação à renda ({loan_to_income:.2f} > 0.3)")
        if taxa_juros > 15:
            print(f"- Taxa de juros está alta ({taxa_juros}% > 15%)")
    else:
        print("Este empréstimo foi classificado como BAIXO RISCO porque:")
        if dti <= 20:
            print(f"- DTI está em nível aceitável ({dti:.1f}% <= 20%)")
        if loan_to_income <= 0.3:
            print(f"- Empréstimo é adequado à renda ({loan_to_income:.2f} <= 0.3)")
        if taxa_juros <= 15:
            print(f"- Taxa de juros está razoável ({taxa_juros}% <= 15%)")

if __name__ == "__main__":
    analisar_novo_emprestimo()
