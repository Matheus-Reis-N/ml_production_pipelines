# scripts/preprocessing.py

def preprocess_data(df):
    """
    Função para pré-processamento específico que seja melhor não deixar por conta do pycaret.

    Args:
        df (DataFrame): Dados a serem pré-processados.

    Returns:
        DataFrame: Dados pré-processados.
    """
    # Aqui você pode adicionar qualquer lógica de pré-processamento, como lidar com valores ausentes, normalização, etc.
    # Exemplo: Remover colunas desnecessárias, imputar valores ausentes diferente das opções do pycaret, etc.
    # df = df.drop(columns=['coluna_desnecessaria'])
    # df = df.fillna(method='ffill')

    print("Dados pré-processados.")
    return df