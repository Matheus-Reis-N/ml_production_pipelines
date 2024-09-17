# %% Imports
import config
import transform_and_load

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import # xyz
# from pycaret.classification import *
# from pycaret.regression import *
import mlflow
import mlflow.pycaret

# %% Configurações MLflow (Tracking, Setting Experiment, Autolog)

# Conectando ao servidor do mlflow da be
mlflow.set_tracking_uri(uri="http://104.237.9.71:5555")

# Conectanto ao experimento
mlflow.set_experiment(experiment_id=366245872357516633) 

# Gerando logs padrão do modelo de forma automática, logs expecíficos ficam à parte
mlflow.autolog()

# %% split,, train,
def split_train_test(df):

    "Separação treino e teste"
    
    X = df.columns[:-1]
    y = df.columns[-1]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size==0.3, random_state=42)

    return x_train, x_test, y_train, y_test

def train_model():
    
    """O setup da run vai:
        -corrigir eventuais datatypes,
        -
    """

    with mlflow.start_run("""
                            Pycaret Clustering classification regression ou timeseries
                            ou Sklearn Clustering classification regression ou timeseries
                          """):
        
        """ o setup do pycaret fará conforme identifique a necessidade:
                - correção de datatypes;
                - tratamento de valores ausentes (preenche por padrão com a média para valores numéricos e moda para categóricos ! - deve ser usado com cautela e verificado se o uso está correto);
                - one-hot encoding para variáveis nominais e ordinal encoding para variáveis ordinais conforme a necessidade de cada modelo;
                - realiza cross-validation por padrão;
                - transformações log, box-cox ou yeo-johnson - tome isto com cautela e sempre verifique se seu uso está correto
        """
        base_analytic = setup(
            data = x_train,
            target = 'target',
            fold_strategy = 'timeseries',
            # normalize = True, # normalização de média 0 e desvio padrão 1. KNN, SVM, etc
            train_size = 0.8,
            log_experiment = True,
            experiment_name = 'unimed_pycaret_matheus',
            log_plots = True,
            # log_profile = True,
            log_data = True,
            data_split_shuffle = False,
            data_split_stratify = False
            sesion_id = 123) # Garantindo a reprodutibilidade

        # em caso de modelo específico em vez de compare_models() use : nome_do_modelo_especifico = create_model(ex.: 'huber')

        # Treinando múltiplos modelos e selecionando o melhor
        best_model = compare_model()

        # Tunando o melhor modelo
        tuned_model = tune_model(best_model)

        # Finalizando o modelo (pronto para produção)
        final_model = finalize_model(tuned_model)
        
        # Salvando modelo final
        save_model(final_model, model_name='../models/xxx')

        # Registrando no MLflow Model Registry
        mlflow.register_model(f"runs:/{mlflow.active_run().info.run_uuid}/xxx", "yyy")
        print(f'Novo modelo salvo em: {mlflow.active_run().info.run_uuid}/xxx')

    mlflow.end_run()

if __name__ == '__main__':
    transformations()
    split_train_test()
    train_model()