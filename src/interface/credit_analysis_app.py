import streamlit as st
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Adiciona o diretório pai ao path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.enhanced_model import treinar_modelo_avancado, analisar_novo_caso

def format_currency(value):
    """Formata valor em reais"""
    return f"R$ {value:,.2f}"

def format_account_time(years):
    """Formata o tempo de conta de forma mais amigável"""
    if years < 1:
        months = int(years * 12)
        return f"{months} meses"
    elif years == 1:
        return "1 ano"
    else:
        return f"{years:.1f} anos"

def calcular_parcela(valor_emprestimo, taxa_juros_anual, prazo_meses):
    """
    Calcula o valor da parcela usando juros compostos
    - valor_emprestimo: valor total do empréstimo
    - taxa_juros_anual: taxa de juros ao ano (em %)
    - prazo_meses: prazo em meses
    """
    taxa_mensal = (1 + taxa_juros_anual/100) ** (1/12) - 1
    parcela = valor_emprestimo * (taxa_mensal * (1 + taxa_mensal) ** prazo_meses) / ((1 + taxa_mensal) ** prazo_meses - 1)
    return parcela

def calcular_custo_total(parcela, prazo_meses, valor_emprestimo):
    """Calcula o custo total do empréstimo"""
    total_pago = parcela * prazo_meses
    juros_pagos = total_pago - valor_emprestimo
    return total_pago, juros_pagos

def ajustar_taxa_juros(valor, prazo_meses, score_credito):
    """Ajusta a taxa de juros baseado no prazo e score de crédito"""
    # Taxa base: 1.5% a.m. = 19.56% a.a.
    taxa_base = 19.56
    
    # Ajuste pelo prazo
    if prazo_meses <= 12:
        ajuste_prazo = 0
    elif prazo_meses <= 24:
        ajuste_prazo = 2
    elif prazo_meses <= 36:
        ajuste_prazo = 4
    else:
        ajuste_prazo = 6
    
    # Ajuste pelo score
    if score_credito >= 800:
        ajuste_score = -4
    elif score_credito >= 700:
        ajuste_score = -2
    elif score_credito >= 600:
        ajuste_score = 0
    elif score_credito >= 500:
        ajuste_score = 2
    else:
        ajuste_score = 4
    
    return taxa_base + ajuste_prazo + ajuste_score

def main():
    # Configuração da página
    st.set_page_config(
        page_title="Sistema de Análise de Crédito",
        layout="wide"
    )
    
    # Título principal
    st.title("Sistema de Análise de Crédito")
    
    # Criar colunas para organizar o layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Dados do Empréstimo")
        with st.expander("Informações do Empréstimo", expanded=True):
            loan_amount = st.number_input(
                "Valor do Empréstimo",
                min_value=1000.0,
                max_value=1000000.0,
                value=20000.0,
                help="Valor solicitado pelo cliente"
            )
            
            # Opções de prazo
            prazo_opcoes = [
                12,    # 1 ano
                18,    # 1.5 anos
                24,    # 2 anos
                36,    # 3 anos
                48,    # 4 anos
                60     # 5 anos
            ]
            
            prazo_index = st.selectbox(
                "Prazo do Empréstimo",
                range(len(prazo_opcoes)),
                format_func=lambda x: f"{prazo_opcoes[x]} meses ({prazo_opcoes[x]/12:.1f} anos)",
                help="Prazo para pagamento do empréstimo"
            )
            prazo_meses = prazo_opcoes[prazo_index]
            
            annual_income = st.number_input(
                "Renda Anual",
                min_value=0.0,
                value=60000.0,
                help="Renda anual total do cliente"
            )
        
        st.header("Histórico de Pagamentos")
        with st.expander("Histórico", expanded=True):
            delay_avg = st.number_input(
                "Média de Dias de Atraso",
                min_value=0.0,
                max_value=90.0,
                value=0.0,
                help="Média de dias de atraso em pagamentos anteriores"
            )
            
            on_time_payments = st.slider(
                "Percentual de Parcelas em Dia",
                min_value=0,
                max_value=100,
                value=100,
                help="Porcentagem de parcelas pagas pontualmente"
            )
            
            max_delay = st.number_input(
                "Maior Atraso Registrado (dias)",
                min_value=0,
                max_value=365,
                value=0,
                help="Maior número de dias de atraso já registrado"
            )
    
    with col2:
        st.header("Relacionamento Bancário")
        with st.expander("Relacionamento com o Banco", expanded=True):
            # Lista de opções de tempo de conta
            tempo_conta_opcoes = [
                0.25,  # 3 meses
                0.5,   # 6 meses
                0.75,  # 9 meses
                1.0,   # 1 ano
                1.5,   # 1 ano e 6 meses
                2.0,   # 2 anos
                3.0,   # 3 anos
                4.0,   # 4 anos
                5.0,   # 5 anos
                7.0,   # 7 anos
                10.0,  # 10 anos
                15.0,  # 15 anos
                20.0   # 20 anos
            ]
            
            tempo_conta_labels = [format_account_time(t) for t in tempo_conta_opcoes]
            
            tempo_conta_index = st.selectbox(
                "Tempo de Conta",
                range(len(tempo_conta_opcoes)),
                format_func=lambda x: tempo_conta_labels[x],
                help="Há quanto tempo o cliente tem conta no banco"
            )
            
            account_time = tempo_conta_opcoes[tempo_conta_index]
            
            avg_balance = st.number_input(
                "Saldo Médio",
                value=5000.0,
                help="Saldo médio mantido na conta nos últimos 3 meses"
            )
            
            uses_overdraft = st.checkbox(
                "Utiliza Cheque Especial",
                help="Marque se o cliente utiliza cheque especial"
            )
        
        st.header("Dados de Crédito")
        with st.expander("Informações de Crédito", expanded=True):
            credit_score = st.slider(
                "Score de Crédito",
                min_value=0,
                max_value=1000,
                value=700,
                help="Pontuação de crédito do cliente (0-1000)"
            )
            
            cpf_queries = st.number_input(
                "Consultas ao CPF (últimos 6 meses)",
                min_value=0,
                max_value=50,
                value=0,
                help="Número de consultas ao CPF nos últimos 6 meses"
            )
            
            has_other_loans = st.checkbox(
                "Possui Outros Empréstimos",
                help="Marque se o cliente possui outros empréstimos ativos"
            )
    
    # Calcular taxa de juros ajustada
    taxa_juros = ajustar_taxa_juros(loan_amount, prazo_meses, credit_score)
    
    # Mostrar simulação antes da análise
    st.header("Simulação do Empréstimo")
    sim_col1, sim_col2, sim_col3 = st.columns(3)
    
    # Calcular valores
    parcela = calcular_parcela(loan_amount, taxa_juros, prazo_meses)
    total_pago, juros_pagos = calcular_custo_total(parcela, prazo_meses, loan_amount)
    
    with sim_col1:
        st.metric(
            "Taxa de Juros",
            f"{taxa_juros:.2f}% a.a.",
            help=f"Taxa ajustada pelo prazo e score de crédito"
        )
    
    with sim_col2:
        st.metric(
            "Valor da Parcela",
            format_currency(parcela),
            help=f"Parcela mensal fixa"
        )
    
    with sim_col3:
        st.metric(
            "Total de Juros",
            format_currency(juros_pagos),
            help=f"Total de juros a serem pagos"
        )
    
    # Mostrar tabela de evolução
    if st.checkbox("Ver Evolução do Empréstimo"):
        evolucao = []
        saldo_devedor = loan_amount
        taxa_mensal = (1 + taxa_juros/100) ** (1/12) - 1
        
        for mes in range(1, prazo_meses + 1):
            juros_mes = saldo_devedor * taxa_mensal
            amortizacao = parcela - juros_mes
            saldo_devedor -= amortizacao
            
            evolucao.append({
                'Mês': mes,
                'Prestação': parcela,
                'Amortização': amortizacao,
                'Juros': juros_mes,
                'Saldo Devedor': max(0, saldo_devedor)
            })
        
        df_evolucao = pd.DataFrame(evolucao)
        st.dataframe(
            df_evolucao.style.format({
                'Prestação': 'R$ {:.2f}',
                'Amortização': 'R$ {:.2f}',
                'Juros': 'R$ {:.2f}',
                'Saldo Devedor': 'R$ {:.2f}'
            })
        )
    
    # Botão de análise
    if st.button("Analisar Crédito", type="primary"):
        with st.spinner("Analisando dados..."):
            # Preparar dados
            dados_cliente = {
                'loan_amnt': loan_amount,
                'int_rate': taxa_juros,
                'annual_inc': annual_income,
                'dias_atraso_media': delay_avg,
                'parcelas_pagas_pontualmente': on_time_payments / 100,
                'maior_atraso': max_delay,
                'tempo_conta_anos': account_time,
                'saldo_medio': avg_balance,
                'usa_cheque_especial': 1 if uses_overdraft else 0,
                'score_credito': credit_score,
                'consultas_cpf_6m': cpf_queries,
                'tem_outros_emprestimos': 1 if has_other_loans else 0
            }
            
            # Fazer análise
            model, feature_names = treinar_modelo_avancado()
            resultado = analisar_novo_caso(model, feature_names, dados_cliente)
            
            # Mostrar resultados
            st.header("Resultado da Análise")
            
            # Criar colunas para o resultado
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                st.metric(
                    "Previsão",
                    resultado['previsao'],
                    delta="Favorável" if resultado['previsao'] == 'BAIXO RISCO' else "Desfavorável"
                )
            
            with res_col2:
                st.metric(
                    "Probabilidade de Default",
                    f"{resultado['probabilidade_default']:.1f}%"
                )
            
            with res_col3:
                # Calcular comprometimento de renda
                renda_mensal = annual_income / 12
                comprometimento = (parcela / renda_mensal) * 100
                st.metric(
                    "Comprometimento de Renda",
                    f"{comprometimento:.1f}%",
                    delta="Alto" if comprometimento > 30 else "Adequado",
                    delta_color="inverse"
                )
            
            # Mostrar fatores de risco
            if resultado['fatores_risco']:
                st.subheader("Fatores de Risco Identificados")
                for fator in resultado['fatores_risco']:
                    st.warning(fator)
            else:
                st.success("Nenhum fator de risco significativo identificado")
            
            # Análise financeira detalhada
            st.subheader("Análise Financeira")
            fin_col1, fin_col2 = st.columns(2)
            
            with fin_col1:
                st.info(f"Renda Mensal: {format_currency(renda_mensal)}")
                st.info(f"Parcela: {format_currency(parcela)}")
            
            with fin_col2:
                st.info(f"Total do Empréstimo: {format_currency(total_pago)}")
                st.info(f"Total de Juros: {format_currency(juros_pagos)}")

if __name__ == "__main__":
    main()
