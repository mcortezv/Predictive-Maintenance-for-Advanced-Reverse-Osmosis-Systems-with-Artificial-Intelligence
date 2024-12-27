import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from model_membrane_condition import modelMembraneCondition
from model_final_tds import ModelFinalTds
from partitions_validation import partitionsValidation

def main():
    
    # Importar Datos Osmosis Inversa
    data = pd.read_csv("model/data/reverse_osmosis_data.csv")

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
    
if __name__ == "__main__":
    main()
    
