import numpy as np

def calcular_taxa_juros(score, prazo):
    """
    Calcula a taxa de juros baseada no score de crédito e prazo do empréstimo
    """
    # Taxa base
    taxa_base = 19.56  # % ao ano
    
    # Ajuste por prazo
    if prazo <= 12:
        ajuste_prazo = 0
    elif prazo <= 24:
        ajuste_prazo = 2
    elif prazo <= 36:
        ajuste_prazo = 4
    else:
        ajuste_prazo = 6
    
    # Ajuste por score
    if score >= 800:
        ajuste_score = -4
    elif score >= 700:
        ajuste_score = -2
    elif score >= 600:
        ajuste_score = 0
    elif score >= 500:
        ajuste_score = 2
    else:
        ajuste_score = 4
    
    # Taxa final
    taxa_final = taxa_base + ajuste_prazo + ajuste_score
    
    return taxa_final