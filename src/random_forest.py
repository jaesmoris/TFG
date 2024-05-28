from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from time import time
import os
import pickle
import joblib

# Funcion para cargar el dataset en el formato Dataset/Experimento/Clase/Partición
def load_dataset(dataset_path):
    classes = os.listdir(dataset_path)
    partition_names = ['train', 'val', 'test']
    partitions = [[], [], []]
    labels = [[], [], []]
    for class_index, class_path in enumerate(classes):
        for partition_index, partition in enumerate(partition_names):
            in_class_paths = os.listdir(dataset_path + "/" + class_path + "/" + partition)
            in_class_paths.sort()
            # For every file in the class directory
            for path in in_class_paths:
                pkl_path = dataset_path + "/" + class_path + "/" + partition + "/" + path
                with open(pkl_path, 'rb') as pkl_file:
                    pkl_object = pickle.load(pkl_file)
                partitions[partition_index].append(pkl_object)
                labels[partition_index].append(class_path)

    return partitions, labels

def load_dataset_by_folds(dataset_path):
    classes = os.listdir(dataset_path)
    partition_names = ['train', 'val', 'test']
    data_folds = [[], [], [], [], []]
    labels_folds = [[], [], [], [], []]
    for class_index, class_path in enumerate(classes):
        for partition_index, partition in enumerate(partition_names):
            in_class_paths = os.listdir(dataset_path + "/" + class_path + "/" + partition)
            in_class_paths.sort()
            # For every file in the class directory
            for path in in_class_paths:
                pkl_path = dataset_path + "/" + class_path + "/" + partition + "/" + path
                with open(pkl_path, 'rb') as pkl_file:
                    pkl_object = pickle.load(pkl_file)
                fold_index = int(path[1])
                data_folds[fold_index].append(pkl_object)
                labels_folds[fold_index].append(class_path)

    return data_folds, labels_folds

def log_file_begin(log_path, experiment_name=""):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as file:
        file.write(f"RANDOM FOREST: {experiment_name}\n\n")

def log_file_string(log_path, string):
    with open(log_path, 'a') as file:
        file.write(string)

experiments = ["Bere", "Landraces", "Origins", "Rows"]
for experiment in experiments:
    dataset_path = f"../../data/Oriented_Divided_SH_L50/{experiment}"
    cv = 4
    random_state = 42

    t_init = time()
    log_path = f"./logs/random_forest_{experiment}.txt"
    log_file_begin(log_path, experiment_name=experiment)

    log_file_string(log_path, f"DATA:\n\n")
    data, labels = load_dataset(dataset_path)
    # Train
    x_train = data[0]
    y_train = labels[0]
    log_file_string(log_path, f"Train:\t{len(x_train)} samples.\n")
    # Val
    x_val = data[1]
    y_val = labels[1]
    log_file_string(log_path, f"Val:\t{len(x_val)} samples.\n")
    # Test
    x_test = data[2]
    y_test = labels[2]
    log_file_string(log_path, f"Test:\t{len(x_test)} samples.\n")

    # Juntar train y val por la cross validation
    x_train = x_train + x_val
    y_train = y_train + y_val
    log_file_string(log_path, f"\nTrain and validation merged to use cross validation.\n")
    log_file_string(log_path, f"Merged train:\t{len(x_train)} samples.\n")

    log_file_string(log_path, f"\nRESULTS:\n\n")


    # Crear el modelo de Random Forest
    model = RandomForestClassifier(random_state=random_state)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    
    # Realizar la búsqueda de cuadrícula con validación cruzada
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=3, scoring='accuracy', error_score='raise')
    grid_search.fit(x_train, y_train)

    # Mejor modelo encontrado
    best_model = grid_search.best_estimator_

    # Evaluar el mejor modelo en el conjunto de prueba
    y_test_pred = best_model.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Calcular la matriz de confusión
    conf_matrix = confusion_matrix(y_test, y_test_pred)

    t_end = time()

    log_file_string(log_path, f'Test Accuracy: {test_accuracy}\n')
    log_file_string(log_path, 'Test Classification Report:\n')
    log_file_string(log_path, classification_report(y_test, y_test_pred))
    log_file_string(log_path,("\nConfusion matrix:\n"))
    log_file_string(log_path,(str(conf_matrix)))
    log_file_string(log_path, "\n\nGrid of parameters:\n")
    log_file_string(log_path, str(param_grid))
    log_file_string(log_path, "\nBest parameters:\n")
    log_file_string(log_path, str(grid_search.best_params_))
    log_file_string(log_path, f"\n\nRandom state: {random_state}.\n")
    log_file_string(log_path, f"Used {cv} folds in Cross Validation.\n")
    log_file_string(log_path, f"Running time: {(t_end - t_init):0.2f} seconds.\n")


    # Save model
    joblib.dump(best_model, f'best_random_forest_model_{experiment}.pkl')
