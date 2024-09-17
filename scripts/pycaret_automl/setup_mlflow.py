import mlflow

def setup_mlflow():
    """
    Conectando ao servidor Mlflow.
    """ 
    mlflow.set_tracking_uri(uri="http://104.237.9.71:5555")

tags = {
        'Versão': '1.0',
        'Time': 'Time de Dados da yyy',
        'Dataset': 'xxx'
        }