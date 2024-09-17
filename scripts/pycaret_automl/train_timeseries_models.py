import mlflow
from pycaret.time_series import setup, compare_models, tune_model, finalize_model, save_model

def train_timeseries_model(x_train, y_train, tags):
    """
    Treina modelos de séries temporais para prever alguma variável contínua.
    Args:
    - df (pd.DataFrame): DataFrame contendo os dados pré-processados.
***** y_train: serie temporal para ser prevista
    - tags (dict): Sobe tags do modelo no Mlflow
    """
    # Criando o experimento
    mlflow.set_experiment('unimed_timeseries_matheus')

    # Rodando
    with mlflow.start_run(run_name='pycaret_timeseries'):
          
        # Registrando as tags no servidor
        mlflow.set_tags(tags)

        # Configuração inicial do PyCaret para séries temporais
        ts = setup(
            data=x_train,
            target=y_train, 
            session_id=42,  # Garantindo Auditabilidade
            n_jobs=-1
            )
        
        # Comparando e treinando modelos de séries temporais
        best_model = compare_models()
        
        # Salvando o modelo com o MLflow
        mlflow.pycaret.log_model(best_model, "best_pycaret_timeseries_model")

    return best_model