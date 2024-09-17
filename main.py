import os
import mlflow
import pycaret
from sklearn.model_selection import train_test_split

from scripts.pycaret_automl.setup_mlflow import setup_mlflow, tags
from scripts.load_data.load_data_local import load_data_from_local
from scripts.load_data.load_data_sharepoint import load_data_from_sharepoint
from scripts.pycaret_automl.preprocessing import preprocess_data
from scripts.pycaret_automl.train_classifier_models import train_classifier_model
from scripts.pycaret_automl.train_clustering_model import train_clustering_model
from scripts.pycaret_automl.train_timeseries_models import train_timeseries_model
from scripts.pycaret_automl.predict import predict


# Escolha a fonte de dados
data_source = 'local'  # Retire o 'local' para carregar do SharePoint
# Escolha o tipo de modelagem
model_type = ''  # Opções: 'classification', 'clustering', 'timeseries'
# Indique onde o modelo está armazenado
model_source = '' # Opções: 'be_server', 's3', 'gcp'

def main():
    # Conectando ao servidor Mlflow da BeAnalytic
    setup_mlflow()

    # Carregando os dados
    if data_source == 'local':
        df = load_data_from_local()
    else:
        df = load_data_from_sharepoint()

    # Verificando se os dados foram carregados corretamente
    if df is not None:
        print("Dados carregados com sucesso!")

        # Separação treino e teste
        train, test = train_test_split(df, test_size=0.25, random_state=42)
        x_train = train.drop('target', axis=1)
        y_train = train['target']
        x_test = test.drop('target', axis=1)
        y_test = test['target']
        print("Split treino e teste realizado com sucesso!")

        # Treinando o modelo baseado no tipo escolhido
        if model_type == 'classification':
            train_classifier_model(x_train, y_train, tags)
        elif model_type == 'clustering':
            train_clustering_model(df, tags)
        elif model_type == 'timeseries':
            train_timeseries_model(x_train, y_train, tags)
        else:
            print("Tipo de modelo inválido.")
            return None
    else:
        print("Falha ao carregar os dados.")

    # Predição de novos dados
    predict(x_test, model_source)
    print(f"Predição Realizada. Pipeline completo executado com sucesso.")

"""
se for clustering voce vai ter que imputar uma nova coluna no dataset chamada cluster para identificar cada instancia ao cluster 
o assign_model faz isso, como se estivessse usando o transform do pandas

kmeans_cluster = assign_model(kmeans)
"""


if __name__ == "__main__":
    main()