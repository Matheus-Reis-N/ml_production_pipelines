import mlflow
from pycaret.clustering import *

def train_clustering_model(df, tags):
    """
    Treina modelos não supervisionado de clustering usando PyCaret.
    Args:
    - df (pd.DataFrame): DataFrame contendo os dados pré-processados.
    - tags (dict): Sobe tags do experimento no Mlflow

    Return: None
    """
    # Conectando ao servidor Mlflow da BeAnalytic
    mlflow.set_tracking_uri(uri="http://104.237.9.71:5555")

    # Rodando
    with mlflow.start_run():
        # Registrando as tags no servidor
        mlflow.set_tags(tags)

        # Configurando o ambiente de PyCaret com integração MLflow
        exp_clusters = setup(
            data=df,
            target=None, # Não supervisionado, sem rótulo
            session_id=123, # Garantindo Auditabilidade
            normalize=True, # Ativando normalização para modelos que precisem
            fold_strategy='kfold', # Kfold para cross-validation
            fold=5,
            train_size=0.8,
            data_split_shuffle=True, # Embaralha os dados antes de dividir
            ## Experiment Logging :
            log_experiment=True, # Gerando relatório de saída para conferir as transformações realizadas automaticamente no pycaret (normalização, imputação de valores ausentes, encoders, etc)
            experiment_name="unimed_clustering_matheus", # Criando e nomeando o experimento
            log_plots=True, # Loga gráficos de avaliação dos modelos
            log_profile=True, # Loga o perfil de dados
            log_data=True # Loga os dados usados para treino
            )

        # Modelos disponíveis para este dataset 
        #models().index

        # Plots disponíveis 
        #help(plot_model)

        # Treinando modelo
        kmeans = create_model(model='kmeans', num_clusters=4) # num_clusters são por padrão 4, nunca use o padrão sempre avalie cada caso
        
        # Salvando o modelo final localmente
        save_model(final_model, model_name='../models/unimed_pycaret_clustering_model')

        # Logando o modelo no MLflow
        log_model(final_model, "model") # Capturando automaticamente parâmetros, métricas e artefatos e subindo no MLflow

        print("Modelo treinado, ajustado e salvo com sucesso.")

    # Fechando o run
    mlflow.end_run()