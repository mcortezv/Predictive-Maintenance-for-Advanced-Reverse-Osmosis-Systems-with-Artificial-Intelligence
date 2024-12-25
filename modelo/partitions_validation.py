
def partitionsValidation(X, Y_membrane_condition, Y_final_tds, X_train, Y_train_membrane, Y_train_tds, 
                         X_val, Y_val_membrane, Y_val_tds, X_test, Y_test_membrane, Y_test_tds):
    
    # Verificación de Particiones
    print("Tamaños: ")
    print("\tDataset Original: {} / Membrane Condition: {} / Final TDS: {}".format(X.shape, Y_membrane_condition.shape, Y_final_tds.shape) )
    print("\tEntrenamiento: {} / Membrane Condition: {} / Final TDS: {}".format(X_train.shape, Y_train_membrane.shape, Y_train_tds.shape) )
    print("\tValidación: {} / Membrane Condition: {} / Final TDS: {}".format(X_val.shape, Y_val_membrane.shape, Y_val_tds.shape) )
    print("\tPrueba: {} / Membrane Condition: {} / Final TDS: {}".format(X_test.shape, Y_test_membrane.shape, Y_test_tds.shape))

    # Proporciones
    print("\nProporciones Categorías (Membrane Condition / Final TDS):\n\nDataset Original:\n")
    dataset_og_membrane_condition = Y_membrane_condition.value_counts(normalize=True)
    dataset_og_final_tds = Y_final_tds.describe()
    print(f"{dataset_og_membrane_condition}")
    print(f"\nTDS Final: \t\n{dataset_og_final_tds}")

    # Entrenamiento (60%)
    dataset_train_membrane_condition = Y_train_membrane.value_counts(normalize=True)
    dataset_train_final_tds = Y_train_tds.describe()
    print(f"\nEntrenamiento: \n\n{dataset_train_membrane_condition}")
    print(f'\nFinal TDS: \n{dataset_train_final_tds}')

    # Validación (20%)
    dataset_val_membrane_condition = Y_val_membrane.value_counts(normalize=True)
    dataset_val_final_tds = Y_val_tds.describe()
    print(f"\nValidación: \n\n{dataset_val_membrane_condition}")
    print(f"\nFinal TDS: \n{dataset_val_final_tds}")

    # Prueba (20%)
    dataset_test_membrane_condition = Y_test_membrane.value_counts(normalize=True)
    dataset_test_final_tds = Y_test_tds.describe()
    print(f"\nPrueba: \n\n{dataset_test_membrane_condition}")
    print(f"\nFinal TDS: \n{dataset_test_final_tds}")
    
    
