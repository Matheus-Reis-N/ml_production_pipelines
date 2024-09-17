from office365.runtime.auth.client_credential import ClientCredential
from office365.sharepoint.client_context import ClientContext
import pandas as pd
import io

# Configurações do SharePoint
site_url = 'https://beanalyticcombr.sharepoint.com'
relative_url = '/caminho/para/o/arquivo/parquet/pacientes_data.parquet'
client_id = 'o_client_id'
client_secret = 'o_client_secret'

def load_data_from_sharepoint(
        site_url=site_url,
        relative_url=relative_url,
        client_id=client_id, 
        client_secret=client_secret):
    """
    Carrega dados do arquivo Parquet armazenado no SharePoint.
    
    Args:
    - site_url (str): URL do site SharePoint.
    - relative_url (str): Caminho relativo do arquivo Parquet no SharePoint.
    - client_id (str): Client ID para autenticação.
    - client_secret (str): Client Secret para autenticação.
    
    Returns:
    - pd.DataFrame: Dados carregados em um DataFrame.
    """
    try:
        # Autenticando no SharePoint
        credentials = ClientCredential(client_id, client_secret)
        context = ClientContext(site_url).with_credentials(credentials)
        
        # Carregando o arquivo do SharePoint
        response = context.web.get_file_by_server_relative_url(relative_url).download(io.BytesIO()).execute_query()
        bytes_file_obj = io.BytesIO(response.content)
        df = pd.read_parquet(bytes_file_obj)
        
        print("Dados carregados com sucesso do SharePoint.")
        return df
    except Exception as e:
        print(f"Erro ao carregar os dados do SharePoint: {e}")
        return None