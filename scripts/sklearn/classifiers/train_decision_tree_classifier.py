from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

def train_decision_tree_clf_with_randomgridsearch(x_train, y_train, x_test, tags):
    """
    Treina um modelo de Árvore de Decisão e exibe o melhor hiperparâmetro e o melhor score.
    
    Args:
    df: DataFrame contendo as features e o target
    target: Nome da coluna alvo (variável dependente)
    tags (dict): Sobe tags do experimento no Mlflow
    """

    # Definindo o espaço de hiperparâmetros para DecisionTree
    param_dist = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    # Definindo a Árvore de Decisão
    clf_dt = DecisionTreeClassifier(random_state=42)

    # Definindo o RandomizedSearchCV
    randomized_search = RandomizedSearchCV(estimator=clf_dt,
                                           param_distributions=param_dist,
                                           n_iter=10,  # Número de combinações a testar
                                           # cv=10,    # Cross Validation, por default = 5
                                           random_state=42,
                                           n_jobs=-1)

    # Ajustando o modelo com RandomizedSearchCV
    randomized_search.fit(x_train, y_train)

    best_model = randomized_search.best_estimator_
    y_pred = best_model.predict(x_test)

    print(f'Decision Tree Classifier:\nMelhor Parâmetro: {randomized_search.best_params_}\nMelhor Score: {randomized_search.best_score_:.2f}')