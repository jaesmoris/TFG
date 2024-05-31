from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from time import time
import os
import pickle
import joblib


def load_dataset_by_folds(dataset_path):
    """Loads a dataset in format Experiment/Class/Partition and returns it by folds."""
    # List classes
    classes = os.listdir(dataset_path)
    partition_names = ['train', 'val', 'test']
    data_folds = [[], [], [], [], []]
    labels_folds = [[], [], [], [], []]
    # For every class
    for class_index, class_path in enumerate(classes):
        # For every partition
        for partition_index, partition in enumerate(partition_names):
            # List folder content
            in_class_paths = os.listdir(dataset_path + "/" + class_path + "/" + partition)
            in_class_paths.sort()
            # For every element
            for path in in_class_paths:
                # Define element path
                pkl_path = dataset_path + "/" + class_path + "/" + partition + "/" + path
                # Load element
                with open(pkl_path, 'rb') as pkl_file:
                    pkl_object = pickle.load(pkl_file)
                # Add element to the correspondent fold
                fold_index = int(path[1]) - 1  # Fold numeration ranges 1-5
                data_folds[fold_index].append(pkl_object)
                labels_folds[fold_index].append(class_path)

    return data_folds, labels_folds

def train_test_split(dataset_path, log_path):
    """Merges 4 folds for cross validation. Returns train(cv) with its split indices and test data."""
    log_file_string(log_path, f"DATA:\n\n")
    data_folds, label_folds = load_dataset_by_folds(dataset_path)
    # Train/val 
    x_train = data_folds[0]  + data_folds[1]  + data_folds[2]
    y_train = label_folds[0] + label_folds[1] + label_folds[2]
    log_file_string(log_path, f"Train:\t{len(y_train)} samples.\n")

    # Validation
    x_val = data_folds[3]
    y_val = label_folds[3]
    log_file_string(log_path, f"Validation:\t{len(y_val)} samples.\n")

    # Test
    x_test = data_folds[4]
    y_test = label_folds[4]
    log_file_string(log_path, f"Test:\t{len(y_test)} samples.\n")
    
    x = x_train + x_val
    y = y_train + y_val
    # Indices list to split into custom folds
    fold_size = len(y_test)
    split_indices = [-1 if i < 3*fold_size else 0 for i in range(len(x))]
    return x, y, x_test, y_test, split_indices

def log_file_begin(log_path, header):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as file:
        file.write(f"{header}\n\n")

def log_file_string(log_path, string):
    with open(log_path, 'a') as file:
        file.write(string)

def train_model(model, param_grid, x, y, split_indices, random_state):
    # Use the list to create PredefinedSplit
    pds = PredefinedSplit(test_fold = split_indices)
    # Search best parameters
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=pds, n_jobs=-1, verbose=3, scoring='accuracy', error_score='raise')
    grid_search.fit(x, y)
    best_model = grid_search.best_estimator_
    return best_model, grid_search
    
def evaluate_model(model, x_test, y_test, log_path=None):
    y_test_pred = model.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    if log_path:
        log_file_string(log_path, f"\nRESULTS:\n\n")
        log_file_string(log_path, f'Test Accuracy: {test_accuracy}\n')
        log_file_string(log_path, 'Test Classification Report:\n')
        log_file_string(log_path, classification_report(y_test, y_test_pred))
        log_file_string(log_path,("\nConfusion matrix:\n"))
        log_file_string(log_path,(str(conf_matrix)))

def print_parameters(log_path, param_grid, best_model_parameters, random_state, time):
    log_file_string(log_path, "\n\nGrid to search best model parameters:\n")
    log_file_string(log_path, "".join([f"\t{key}: {value}\n" for key, value in param_grid.items()]))
    log_file_string(log_path, "\nBest combination of parameters:\n")
    log_file_string(log_path, "".join([f"\t{key}: {value}\n" for key, value in grid_search.best_params_.items()]))
    log_file_string(log_path, "\nModel parameters:\n")
    log_file_string(log_path, "".join([f"\t{key}: {value}\n" for key, value in best_model_parameters.items()]))
    log_file_string(log_path, f"\n\nRandom state: {random_state}.\n")
    log_file_string(log_path, f"Used custom folds in Cross Validation.\n")
    log_file_string(log_path, f"Running time: {time:0.2f} seconds.\n")

    
random_state = 42
model_prefix = "RF"
experiments = ["Landraces", "Bere", "Origins", "Rows"]
for experiment in experiments:
    t_init = time()
    
    # Paths
    # dataset_path = f"../../data/Oriented_Divided_SH_L50/{experiment}" # local
    dataset_path = f"/home/msiau/data/tmp/jesmoris/Oriented_Divided_SH_L50/{experiment}" # CVC server
    log_path = f"../logs/{model_prefix}/no_cv/{model_prefix}_{experiment}.txt"
    save_model_path = f"../models/{model_prefix}/no_cv/{model_prefix}_{experiment}.pkl"
    
    # Make folders to save logs and models
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file_begin(log_path, f"{model_prefix} without cross validation: {experiment}")

    # Load data
    x, y, x_test, y_test, split_indices = train_test_split(dataset_path, log_path)

    """
    # Model definition MLP
    model = MLPClassifier(max_iter=300, random_state=random_state)
    param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100), (100, 50, 25), (200, 100, 50)],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'learning_rate_init': [0.001, 0.01, 0.1]
    }
    """
    # Model definition RF
    model = RandomForestClassifier(random_state=random_state)

    param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_features': ['sqrt', 'log2', 0.5],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5]
    }
    
    # Find and evaluate best model
    best_model, grid_search = train_model(model, param_grid, x, y, split_indices, random_state)
    evaluate_model(best_model, x_test, y_test, log_path=log_path)
    
    # Save model
    joblib.dump(best_model, save_model_path)
    
    t_end = time()
    print_parameters(log_path, param_grid, best_model.get_params(), random_state, (t_end - t_init))
