from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

def train_logreg_clf_with_randomgridsearch(x_train, y_train, x_test, tags):
    """
    Treina um modelo de Regressão Logística e exibe o melhor hiperparâmetro e o melhor score.
    
    Args:
    df: DataFrame contendo as features e o target
    target: Nome da coluna alvo (variável dependente)
    tags (dict): Sobe tags do experimento no Mlflow
    """

    # Definindo o espaço de hiperparâmetros para Logistic Regression
    param_dist = {
        'C': np.logspace(-4, 4, 20),                            # Valores para deixar a regressão mais suave
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],          # Tipos de penalização
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga'],  # Métodos/formas de construção da regressão logística
        'max_iter': [100, 200, 500]                             # Número máximo de iterações
    }

    # Definindo a Regressão Logística
    clf_lr = LogisticRegression(random_state=42)

    # Definindo o RandomizedSearchCV
    randomized_search = RandomizedSearchCV(estimator=clf_lr,
                                           param_distributions=param_dist,
                                           n_iter=10,  # Número de combinações a testar
                                           # cv=10,    # Cross Validation, por default = 5
                                           random_state=42,
                                           n_jobs=-1)

    # Ajustando o modelo com RandomizedSearchCV
    randomized_search.fit(x_train, y_train)

    best_model = randomized_search.best_estimator_
    y_pred = best_model.predict(x_test)

    print(f'Logistic Regression:\nMelhor Parâmetro: {randomized_search.best_params_}\nMelhor Score: {randomized_search.best_score_:.2f}')