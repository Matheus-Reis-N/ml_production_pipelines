from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

def train_random_forest_clf_with_randomgridsearch(x_train, y_train, x_test, tags):
    """
    Treina 50 modelos de Random Forest (n_iter * cv) e escolhe o melhor
    
    - x_train (numpy.ndarray): Features (variáveis independentes) do DataFrame separadas para treino.
    - y_train (numpy.ndarray): Nome da coluna alvo para previsão (variável dependentes).
    - x_test (numpy.ndarray): Features (variáveis independentes) do DataFrame separadas para teste
    - tags (dict): Sobe tags do experimento no Mlflow
        
    Return: None
    """

    # Definindo o espaço de hiperparâmetros para o Random Forest Classifier
    param_dist = {
        'n_estimators': [100, 200, 500], 
        'max_depth': [None, 10, 20, 30]
        }

    # Definindo o classificador Random Forest
    clf_rf = RandomForestClassifier(random_state=42)

    # Definindo o RandomizedSearchCV
    randomized_search = RandomizedSearchCV(estimator=clf_rf,
                                           param_distributions=param_dist,
                                           #n_iter=10,  # Número de combinações de parâmetros a testar, por default = 10. Intervalos médios vão de 10-50, Note que n_iter precisa ser necessariamente menor que as combinações dos parâmetros no param_dist
                                           #cv=10,      # Cross Validation, por default = 5
                                           #refit=True  # Treinando o modelo final com todo o dataset e hiperparametros tunados, por deafult = True
                                           random_state=42,
                                           n_jobs=-1)
    
    # Treinando modelos, afinando hiperparametros e escolhendo o melhor tune encontrado com RandomizedSearchCV
    randomized_search.fit(x_train, y_train)

    # Melhor modelo encontrado
    best_model = randomized_search.best_estimator_
    
    # Salvando o modelo final localmente (persistindo) como .pkl

    # Deploying model on Model Registry

    #y_pred = best_model.predict(x_test)
    #randomized_search.best_score_

    print(f'Hiperparâmetros do modelo Random Forest inicial: {clf_rf.get_params()}\n'
          f'Hiperparâmetros tunados por RandomSearchCV: {best_model.get_params()}'
          )

def train_random_forest_clf_with_outros_otimizadores():
    return None