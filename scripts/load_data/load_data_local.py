import pandas as pd

# Caminho do arquivo de dados local
tbExtractCusto = '../data/tbExtractCusto.parquet'
tbPopulacaoAssoc = '../data/tbPopulacaoAssoc.parquet'
tbPopulacaoTijuco = '../data/tbPopulacaoTijuco.parquet'
tbPopulacaoUfu = '../data/tbPopulacaoUfu.parquet'

file_name = .split('/')[-1].replace('.parquet', '')


def load_data_from_local(file_path=file_path):
    """
    Carrega dados do arquivo Parquet localizado neste sistema de diretórios.
    
    Arg:
    - file_path (str): Caminho para o arquivo Parquet. Se não especificado, 
      utiliza o caminho padrão na pasta 'data' deste sistema de diretórios.
    
    Returns:
    - pd.DataFrame: Dados carregados em um DataFrame.
    """
    try:
        df = pd.read_parquet(file_path)
        print(f"Dados carregados com sucesso de {file_path}.")
        return df
    except Exception as e:
        print(f"Erro ao carregar os dados: {e}")
        return None