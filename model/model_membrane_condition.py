from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

def modelMembraneCondition(X_train_res, Y_train_membrane_res, X_val_s, X_test_s, Y_val_membrane, Y_test_membrane):

    # Parametros Modelo Clasificación (Random Forest Classifier)
    parameters = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    parameters_search = GridSearchCV(RandomForestClassifier(random_state = 42), parameters, cv = 5, scoring = "f1_macro")
    parameters_search.fit(X_train_res, Y_train_membrane_res)

    # Modelo Membrane Condition 
    model_membrane_condition = parameters_search.best_estimator_

    # Predicciones Membrane Condition
    Y_val_pred_membrane = model_membrane_condition.predict(X_val_s)
    Y_test_pred_membrane = model_membrane_condition.predict(X_test_s)

    # Evaluación Modelo Clasificación (Validación)
    print("\n(Validación) Matriz Confusión Membrane Condition:")
    print(confusion_matrix(Y_val_membrane, Y_val_pred_membrane))
    print("\nInforme de Clasificación Membrane Condition (Validación):")
    print(classification_report(Y_val_membrane, Y_val_pred_membrane))

    # Evaluación Modelo Clasificación (Prueba)
    print("(Prueba) Matriz Confusión Membrane Condition:")
    print(confusion_matrix(Y_test_membrane, Y_test_pred_membrane))
    print("\nInforme de Clasificación Membrane Condition (Prueba):")
    print(classification_report(Y_test_membrane, Y_test_pred_membrane))
    
