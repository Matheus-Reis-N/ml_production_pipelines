from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RandomizedSearchCV

def train_dummy_clf_with_randomgridsearch(x_train, y_train, x_test, tags):
    """
    Treina um Dummy Classifier e exibe o melhor hiperparâmetro e o melhor score.
    
    Args:
    df: DataFrame contendo as features e o target
    target: Nome da coluna alvo (variável dependente)
    tags (dict): Sobe tags do experimento no Mlflow
    """

    # Definindo o espaço de hiperparâmetros para o DummyClassifier
    param_dist = {
        'strategy': ['most_frequent', 'stratified', 'uniform']
    }

    # Definindo o classificador Dummy
    clf_dummy = DummyClassifier(random_state=42)

    # Definindo o RandomizedSearchCV
    randomized_search = RandomizedSearchCV(estimator=clf_dummy,
                                           param_distributions=param_dist,
                                           n_iter=10,  # Número de combinações a testar
                                           # cv=10,    # Cross Validation, por default = 5
                                           random_state=42,
                                           n_jobs=-1)

    # Ajustando o modelo com RandomizedSearchCV
    randomized_search.fit(x_train, y_train)

    best_model = randomized_search.best_estimator_
    y_pred = best_model.predict(x_test)

    # Exibindo o melhor score e os parâmetros do melhor modelo
    print(f'Dummy Classifier:\nMelhor Parâmetro: {randomized_search.best_params_}\nMelhor Score: {randomized_search.best_score_:.2f}')