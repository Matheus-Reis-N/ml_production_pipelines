import mlflow
from pycaret.classification import *

def specific_train_model():
    return

def ensemble():
    return

def blend():
    # Treinando modelos --- EXEMPLO
    lr = create_model('lr')
    dt = create_model('dt')
    knn = create_model('knn')

    # Blending
    blender = blend_models([lr, dt, knn])


def train_classifier_model(x_train, y_train, tags):
    """
    Treina modelos supervisionados de classificação usando PyCaret.
    Args:
    - x_train (numpy.ndarray): Features (variáveis independentes) do DataFrame.
    - target (str): Nome da coluna alvo para previsão (variável dependentes).
    - tags (dict): Sobe tags do experimento no Mlflow

    Return: None
    """
    # Rodando
    with mlflow.start_run():
        # Configuração inicial do PyCaret
        print('-> Definindo Setup Pycaret\n\n')
        clf = setup(
            ## Data
            data=x_train,
            target=y_train,
            session_id=123,               # Garantindo Auditabilidade
            ## Feature Engineering 
            normalize=True,               # Ativa normalização para modelos que precisem
            #normalize_method=''          # Por padrão 'zcore' ('zscore', 'minmax', 'maxabs', 'robust')
            #ignore_features='column_x'
            #pca='True',                  # Faz redução de dimensionalidade automaticamente
            #pca_method='',               # Por padrão 'linear' ('linear', 'kernel', 'incremental')
            #fix_imbalance=True,          # Balanceia as classes com SMOTE (Synthetic Minority Over-sampling Technique)
            ## Model Selection
            fold_strategy='kfold',        # ‘kfold’ ‘stratifiedkfold’ ‘groupkfold’ ‘timeseries’ ou outro cross-validation pode ser customizado desde que compatível com o sklearn
            fold=5,                       # Escolhas usuais são 5, 10, 15
            train_size=0.8,
            data_split_shuffle=True,      # Embaralha os dados antes de dividir
            ## Experiment Logging :
            log_experiment=True,          # Gera relatório de saída para conferir as transformações realizadas automaticamente no pycaret (normalização, imputação de valores ausentes, encoders, etc)
            experiment_name="unimed_classification_matheus",
            experiment_custom_tags=tags,
            log_plots=True,               # Loga gráficos de avaliação dos modelos
            log_profile=True,             # Loga o perfil de dados
            log_data=True,                # Loga os dados usados para treino
            ## Controle de uso de Processamento
            #n_jobs=None                  # Por padrão o n_jobs = -1, vai rodar todos os processadores em paralelo. None = roda em somente um
            )
        print('\n\n-> Setup definido !')
        
        # Comparando vários modelos de classificação para selecionar o melhor
        print('\n\n-> Comparando modelos')
        best_model = compare_models(sort='Accuracy', ) # turbo=True, budget_time=120  -- > use o turbo e budget_time conforme o tempo e recursos disponíveis para cada caso
        # tree_models = compare_models(include = ['dt', 'rf', 'et', 'gbc', 'xgboost', 'lightgbm', 'catboost'])
        print(f'\n\n-> Modelo {best_model} escolhido !')

        # Afinando hiperparâmetros e escolhendo o melhor tune encontrado
        print(f'\n\n-> Tunando {best_model}')
        tuned_model, tuner = tune_model(best_model, 
                                 return_tuner=True,   # Retorna o otimizador escolhido, por default : RandomSearchCV
                                 choose_better=True,  # Se o otimizador não otimizar o modelo inicial ele vai retornar o melhor (o inicial, neste caso) 
                                 optimize='Accuracy', # Seleciona o melhor modelo dada a métrica especificada aqui, por deafult = Acurácia. 
                                 n_iter=30)           # Parâmetro n_iter (número de iterações do RandomizedSearchCV) por padrão é 10, ajuste conforme o tempo e recursos disponíveis para cada caso
        print(f'\n\n-> Tunning completo !')
        
        print(f'\n\nHiperparâmetros do modelo inicial: {best_model}\n'
              f'Hiperparâmetros tunados por {type(tuner)}: {tuned_model}')

        print('\n\nTreinando o modelo final com todo o dataset e hiperparâmetros tunados')
        final_model = finalize_model(tuned_model)

        print('\n\nSalvando o modelo final localmente (persistindo) como .pkl na pasta "models"')
        save_model(final_model, model_name='../models/pycaret_classification_model')

"""-----------------------------
        # Deploying model on Model Registry
        mlflow.register_model(f"{mlflow.active_run().info.run_uuid}/unimed_model", "pycaret_classification_model")
        print(f'Novo modelo salvo em: {mlflow.active_run().info.run_uuid}/unimed_model')

        # Deploing model on AWS S3
        deploy_model(
            final_model, 'insurance-pipeline-aws', 
            platform = 'aws',
            authentication = {'bucket' : 'pycaret-test'})
-----------------------------
        from mlflow import MlflowClient

        client = MlflowClient()

        client.
-----------------------------
        #Criar uma def para subir o modelo nas clouds tbm
        

        print("Processo de treinamento completo com sucesso !")"""

    # Fechando o run
    mlflow.end_run()