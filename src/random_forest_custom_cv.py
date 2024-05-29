from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import confusion_matrix
from time import time
import os
import pickle
import joblib

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
                fold_index = int(path[1]) - 1
                data_folds[fold_index].append(pkl_object)
                labels_folds[fold_index].append(class_path)

    return data_folds, labels_folds

def log_file_begin(log_path, experiment_name=""):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as file:
        file.write(f"MULTILAYER PERCEPTRON: {experiment_name}\n\n")

def log_file_string(log_path, string):
    with open(log_path, 'a') as file:
        file.write(string)

random_state = 42
experiments = ["Landraces", "Bere", "Origins", "Rows"]
for experiment in experiments:
    # dataset_path = f"../../data/Oriented_Divided_SH_L50/{experiment}" # local
    dataset_path = f"/home/msiau/data/tmp/jesmoris/Oriented_Divided_SH_L50/{experiment}" # CVC server
    log_path = f"../logs/RF/custom_cv/custom_cv_random_forest_{experiment}.txt"
    save_model_path = f"../models/RF/custom_cv/custom_cv_best_random_forest_model_{experiment}.pkl"
    
    t_init = time()
    log_file_begin(log_path, experiment_name=experiment)

    log_file_string(log_path, f"DATA:\n\n")
    data_folds, label_folds = load_dataset_by_folds(dataset_path)
    
    # Train/val 
    x = data_folds[0]  + data_folds[1]  + data_folds[2] + data_folds[2]
    y = label_folds[0] + label_folds[1] + label_folds[2] + label_folds[3]
    log_file_string(log_path, f"{len(data_folds)-1} folds of {len(data_folds[0])} samples to cross validate, Total:\t{len(x)} samples.\n")
    # Test
    x_test = data_folds[4]
    y_test = label_folds[4]
    log_file_string(log_path, f"Test:\t{len(x_test)} samples.\n")


    log_file_string(log_path, f"\nRESULTS:\n\n")


    # Crear el modelo de Random Forest
    model = RandomForestClassifier(random_state=random_state)

    param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_features': ['sqrt', 'log2', 0.5],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5]
    }
    
    # Create a list where train data indices are -1 and validation data indices are 0
    fold_size = len(y_test)
    split_index = [i // fold_size for i in range(len(x))]
    # print(split_index)

    # Use the list to create PredefinedSplit
    pds = PredefinedSplit(test_fold = split_index)
    
    # Realizar la búsqueda de cuadrícula con validación cruzada
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=pds, n_jobs=-1, verbose=3, scoring='accuracy', error_score='raise')
    grid_search.fit(x, y)

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
    log_file_string(log_path, f"Used custom folds in Cross Validation.\n")
    log_file_string(log_path, f"Running time: {(t_end - t_init):0.2f} seconds.\n")


    # Save model
    joblib.dump(best_model, save_model_path)
