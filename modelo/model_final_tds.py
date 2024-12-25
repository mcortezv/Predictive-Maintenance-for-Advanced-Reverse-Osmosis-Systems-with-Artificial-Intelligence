from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def ModelFinalTds(X_train_s, Y_train_tds, X_val_s, X_test_s, Y_val_tds, Y_test_tds):
    
    # Entrenamiento Modelo Regresión TDS Final (Random Forest Regression)
    model_final_tds = RandomForestRegressor(n_estimators = 100, random_state = 42)
    model_final_tds.fit(X_train_s, Y_train_tds)

    # Predicciones TDS Final
    Y_val_pred_tds = model_final_tds.predict(X_val_s)
    Y_test_pred_tds = model_final_tds.predict(X_test_s)

    # Evaluación Modelo Regresión
    mse_val = mean_squared_error(Y_val_tds, Y_val_pred_tds)
    r2_val = r2_score(Y_val_tds, Y_val_pred_tds)
    mse_test = mean_squared_error(Y_test_tds, Y_test_pred_tds)
    r2_test = r2_score(Y_test_tds, Y_test_pred_tds)
    
    print(f"\n(Validacion) Error Cuadrático Medio y Coeficiente de Determinacion Final TDS \n\tMSE = {mse_val}, R² = {r2_val}")
    print(f"\n(Prueba) Error Cuadrático Medio y Coeficiente de Determinacion Final TDS \n\tMSE = {mse_test}, R² = {r2_test}\n")
    
    
    
