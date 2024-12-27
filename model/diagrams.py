import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from model_membrane_condition import modelMembraneCondition
from model_final_tds import ModelFinalTds
from partitions_validation import partitionsValidation
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # Importar Datos Osmosis Inversa
    data = pd.read_csv("data/reverse_osmosis_data.csv")

    # Variables Predictoras
    X = data[["Feed Pressure (Pa)", "Permeate Flow (L/h)", "Concentrate Flow (L/h)" , "Differential Pressure (bar)",
            "Feed Water Temperature (°C)", "Total Dissolved Solids (ppm)", "Water Conductivity (µS/cm)",
            "Chlorine Concentration (mg/L)", "Total Operating Hours (h)", "Salt Rejection Rate (%)",
            "Fouling Index (SDI)", "Feed Water pH", "Membrane Pore Size (nm)"]]

    # Variables Objetivo
    Y_membrane_condition = data["Membrane Condition"]
    Y_final_tds = data["Final TDS (ppm)"]

    # Entrenamiento (60%) y Temporal de Validación/Prueba (40%)
    X_train, X_temp, Y_train_membrane, Y_temp_membrane, Y_train_tds, Y_temp_tds = train_test_split(X, Y_membrane_condition, Y_final_tds, test_size = 0.4, random_state = 42)

    # Validacion (20%) y Prueba (20%)
    X_val, X_test, Y_val_membrane, Y_test_membrane, Y_val_tds, Y_test_tds = train_test_split(X_temp, Y_temp_membrane, Y_temp_tds, test_size = 0.5, random_state = 42, stratify = Y_temp_membrane)

    # Particiones Entrenamiento (60%), Validacion (20%) y Prueba (20%)
    partitionsValidation(X, Y_membrane_condition, Y_final_tds, X_train, Y_train_membrane, Y_train_tds, 
                         X_val, Y_val_membrane, Y_val_tds, X_test, Y_test_membrane, Y_test_tds)

    # Escalar Datos (Entrenamiento, Validacion y Prueba)
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Balanceo Clases Membrane Condition
    smote = SMOTE(random_state = 42)
    X_train_res, Y_train_membrane_res = smote.fit_resample(X_train_s, Y_train_membrane)

    modelMembraneCondition(X_train_res, Y_train_membrane_res, X_val_s, X_test_s, Y_val_membrane, Y_test_membrane)
    ModelFinalTds(X_train_s, Y_train_tds, X_val_s, X_test_s, Y_val_tds, Y_test_tds)

def modelMembraneCondition(X_train_res, Y_train_membrane_res, X_val_s, X_test_s, Y_val_membrane, Y_test_membrane):
    # Parametros Modelo Clasificación (Random Forest Classifier)
    parameters = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10]
    }
    parameters_search = GridSearchCV(RandomForestClassifier(random_state = 42), parameters, cv = 5, scoring = "f1_macro")
    parameters_search.fit(X_train_res, Y_train_membrane_res)

    # Modelo Membrane Condition 
    model_membrane_condition = parameters_search.best_estimator_

    # Predicciones Membrane Condition
    Y_val_pred_membrane = model_membrane_condition.predict(X_val_s)
    Y_test_pred_membrane = model_membrane_condition.predict(X_test_s)

    # Evaluación Modelo Clasificación (Validación)
    plot_confusion_matrix(Y_val_membrane, Y_val_pred_membrane, "Matriz Confusión (Validación)")
    print("\nInforme de Clasificación Membrane Condition (Validación):")
    print(classification_report(Y_val_membrane, Y_val_pred_membrane))

    # Evaluación Modelo Clasificación (Prueba)
    plot_confusion_matrix(Y_test_membrane, Y_test_pred_membrane, "Matriz Confusión (Prueba)")
    print("\nInforme de Clasificación Membrane Condition (Prueba):")
    print(classification_report(Y_test_membrane, Y_test_pred_membrane))

def ModelFinalTds(X_train_s, Y_train_tds, X_val_s, X_test_s, Y_val_tds, Y_test_tds):
    # Entrenamiento Modelo Regresión TDS Final (Random Forest Regression)
    model_final_tds = RandomForestRegressor(n_estimators=100, random_state=42)
    model_final_tds.fit(X_train_s, Y_train_tds)

    # Predicciones TDS Final
    Y_val_pred_tds = model_final_tds.predict(X_val_s)
    Y_test_pred_tds = model_final_tds.predict(X_test_s)

    # Evaluación Modelo Regresión
    mse_val = mean_squared_error(Y_val_tds, Y_val_pred_tds)
    r2_val = r2_score(Y_val_tds, Y_val_pred_tds)
    mse_test = mean_squared_error(Y_test_tds, Y_test_pred_tds)
    r2_test = r2_score(Y_test_tds, Y_test_pred_tds)
    
    print(f"\n(Validación) Error Cuadrático Medio y Coeficiente de Determinación Final TDS \n\tMSE = {mse_val}, R² = {r2_val}")
    print(f"\n(Prueba) Error Cuadrático Medio y Coeficiente de Determinación Final TDS \n\tMSE = {mse_test}, R² = {r2_test}\n")

    # Gráfica de Predicciones vs Valores Reales
    plt.figure(figsize=(10, 6))
    plt.scatter(Y_val_tds, Y_val_pred_tds, alpha=0.7, label='Validación', color='blue')
    plt.scatter(Y_test_tds, Y_test_pred_tds, alpha=0.7, label='Prueba', color='orange')
    plt.plot([Y_val_tds.min(), Y_val_tds.max()], [Y_val_tds.min(), Y_val_tds.max()], 'k--', lw=2)
    plt.xlabel("Valores Reales")
    plt.ylabel("Predicciones")
    plt.title("Predicciones vs Valores Reales para TDS Final")
    plt.legend()
    plt.grid()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Clase 0', 'Clase 1'], yticklabels=['Clase 0', 'Clase 1'])
    plt.title(title)
    plt.xlabel("Predicción")
    plt.ylabel("Realidad")
    plt.show()

if __name__ == "__main__":
    main()
