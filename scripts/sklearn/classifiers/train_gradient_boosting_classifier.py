from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV

def train_gradient_boosting_clf_with_randomgridsearch(x_train, y_train, x_test, tags):
    """
    Treina um modelo de Gradient Boosting e exibe o melhor hiperparâmetro e o melhor score.
    
    Args:
    df: DataFrame contendo as features e o target
    target: Nome da coluna alvo (variável dependente)
    tags (dict): Sobe tags do experimento no Mlflow
    """

    # Definindo o espaço de hiperparâmetros para o gradiente boosting
    param_dist = {
        'n_estimators': [100, 200, 500], 
        'learning_rate': [0.01, 0.1, 1], 
        'max_depth': [3, 5, 10]
        }

    # Definindo o Gradiente Boosting
    clf_gb = GradientBoostingClassifier(random_state=42)

    # Definindo o RandomizedSearchCV
    randomized_search = RandomizedSearchCV(estimator=clf_gb,
                                           param_distributions=param_dist,
                                           n_iter=10,  # Número de combinações a testar
                                           # cv=10,    # Cross Validation, por default = 5
                                           n_jobs=-1)

    # Ajustando o modelo com RandomizedSearchCV
    randomized_search.fit(x_train, y_train)

    best_model = randomized_search.best_estimator_
    y_pred = best_model.predict(x_test)

    print(f'Gradient Boosting Classifier:\nMelhor Parâmetro: {randomized_search.best_params_}\nMelhor Score: {randomized_search.best_score_:.2f}')