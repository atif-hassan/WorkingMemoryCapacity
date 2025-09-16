import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import warnings
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses



def feature_importance(columns_with_indices, model, X_train, num_repeats, columns_with_importance):
    preds = model.predict_proba(X_train)
    loss1 = log_loss(Y_train, preds)
    # Store the importance scores in columns_with_importance variable
    for column_name, indices in tqdm(columns_with_indices.items(), total=len(columns_with_indices), desc="Finding feature importance"):
        # Perform K-Fold permuted feature importance
        total_imp = list()
        for i in range(num_repeats):
            # Corrupt the copy of the feature
            X_train_shuffled = X_train.copy()
            for i in indices:
                np.random.shuffle(X_train_shuffled[:, i])  # shuffle one feature at a time
            # Get the loss
            preds = model.predict_proba(X_train_shuffled)
            loss2 = log_loss(Y_train, preds)
            # Save the difference in loss values
            total_imp.append(np.abs(loss1-loss2))
        total_imp = np.asarray(total_imp)
        columns_with_importance[column_name].append(np.mean(total_imp))
    return columns_with_importance





if __name__ == "__main__":
    ####################################### SET THIS PARAMETER #############################################
    target_type = "dnb1"
    # Options:
    #   1. dnb1
    #   2. dnb1+dnb2
    num_repeats = 50
    k = 10
    ########################################################################################################
    
    
    
    
    
    ########################################## GET THE TARGET ###############################################
    # First get the target
    target_dir = "target.xlsx"
    target_raw = pd.DataFrame(pd.read_excel(target_dir, engine="calamine"))
    # Convert to a dictionary
    target_raw = target_raw[["Name", "DnB1.1", "DnB1.2", "DnB2.1", "DnB2.2"]].values
    target = dict()
    # Decide the target based on the target type
    for name, score1, score2, score3, score4 in target_raw:
        if target_type == "dnb1":
            target[name.strip().lower()] = 1 if score1 >= 80 else 0
        else:
            target[name.strip().lower()] = 1 if (score1 >= 80 or score3 >= 80) else 0
    #########################################################################################################
    
    
    
    
    
    chunks = {
        "chunks5": 5,
        "chunks10": 10, 
        "chunks15": 15, 
        "chunks20": 20, 
        "chunks25": 25
        }
    for chunk, chunk_val in chunks.items():
        ####################################### TRAIN DATA PREPROCESSING ############################################
        print("\nChunk:", chunk, "\n==========================\n")
        if target_type == "dnb1":
            train_dir = "DnB1/"+chunk+"/train/"
        else:
            train_dir = "DnB1+DnB2/"+chunk+"/train/"
        file_names = [file_name for file_name in os.listdir(train_dir) if file_name.endswith('.xlsx')]
        file_names.remove("tnh_36.xlsx")#
        file_names.remove("tnh_38.xlsx")#
        # Remove this data point when considering DnB2
        if target_type != "dnb1":
            file_names.remove("tnh_67.xlsx")#
        X_train_seq, X_train_static, Y_train = list(), list(), list()
        for file_name in tqdm(file_names):
            # print(file_name)
            df = pd.DataFrame(pd.read_excel(train_dir+file_name, engine="calamine"))
            # Get all the static values
            static_feats = np.concatenate((df[["baseline_pd", "baseline_gsr"]].values[0], df[["gsr_responses"]].mean()))
            # static_feats = df[["baseline_pd"]].values[0]
            X_train_static.append(static_feats)
            # Drop extra columns
            # df = df.drop(["T2LF", "T2FF", "TSD", "total_responses", "TFD", "NOF", "reaction_time (s)", "baseline_gsr", "baseline_pd", "gsr_responses"], axis=1)
            df = df.drop(["T2LF", "T2FF", "TSD", "total_responses", "TFD", "NOF", "reaction_time (s)", "baseline_gsr", "baseline_pd", "gsr_responses", "avg_gsr_amplitude", "gsr_response_proportion"], axis=1)
            # Find the total number of remaining columns
            num_columns = len(df.columns)
            # Since we will be flattening the data, find the indices corresponding to each column
            columns_with_indices = dict([[column_name, np.arange(index, chunk_val*num_columns, num_columns)] for index, column_name in enumerate(df.columns)])
            columns_with_indices["baseline_pd"] = [chunk_val*num_columns]
            columns_with_indices["baseline_gsr"] = [(chunk_val*num_columns) + 1]
            columns_with_indices["mean_gsr_responses"] = [chunk_val*num_columns+2]
            # Impute missing values
            if df.isnull().values.any():
                curr_X = df.values
                # imp_mean = IterativeImputer(RandomForestRegressor(random_state=27), random_state=27)
                imp_mean = IterativeImputer(random_state=27)
                imp_mean.fit(curr_X)
                curr_X = imp_mean.transform(curr_X)
            else:
                curr_X = df.values
            scaler = StandardScaler()
            curr_X = scaler.fit_transform(curr_X)
            # Get the data in correct format for machine learning
            curr_X = curr_X.flatten()
            X_train_seq.append(curr_X)
            curr_Y = target[file_name[:file_name.index(".")]]
            Y_train.append(curr_Y)
        X_train_seq, X_train_static, Y_train = np.asarray(X_train_seq), np.asarray(X_train_static), np.asarray(Y_train)
        # Standardize the static features and concatenate with sequential features
        scaler_static = StandardScaler()
        X_train_static = scaler_static.fit_transform(X_train_static)
        X_train = np.concatenate((X_train_seq, X_train_static), axis=1)
        #########################################################################################################
        
        
        
        
        
        ####################################### FIND BEST HYPER-PARAMETERS #####################################
        # Parameter grid
        param_grid = {
            'n_estimators': np.arange(70, 400, 10),
            'max_depth': [2, 3, 4, 5, 6],
            'min_samples_split': [2, 5, 7, 10],
            'max_features': ['sqrt', 'log2']
        }

        # Perform grid search to find the best hyper-parameter combination
        kfold = StratifiedKFold(n_splits=k, random_state=27, shuffle=True)
        # Initialize and fit GridSearch
        grid_search = GridSearchCV(RandomForestClassifier(random_state=27), param_grid, cv=kfold.split(X_train, Y_train), error_score="raise", n_jobs=-1, verbose=0)
        grid_search.fit(X_train, Y_train)
        # Best Parameters
        print("Best Parameters:", grid_search.best_params_)
        #########################################################################################################
        
        
        
        
        ####################################### PERFORM K-FOLD CROSS-VALIDATION #####################################
        # Create a dictionary containing feature names and corresponding importance scores
        columns_with_importance = dict([[column_name, list()] for column_name in columns_with_indices.keys()])
        # Now check the results
        total_acc = list()
        kfold = StratifiedKFold(n_splits=k, random_state=27, shuffle=True)
        for idx, (train_idx, test_idx) in enumerate(kfold.split(X_train, Y_train)):
            x_train, x_val = X_train[train_idx], X_train[test_idx]
            y_train, y_val = Y_train[train_idx], Y_train[test_idx]
            
            model = RandomForestClassifier(n_estimators=grid_search.best_params_["n_estimators"], 
                                        max_depth=grid_search.best_params_["max_depth"],
                                        min_samples_split=grid_search.best_params_["min_samples_split"],
                                        max_features=grid_search.best_params_["max_features"],
                                        random_state=27)
            model.fit(x_train, y_train)
            preds = model.predict(x_val)
            acc = accuracy_score(y_val, preds)
            print("Acc in fold", idx+1, ":", acc)
            total_acc.append(acc)
            # Calculate feature importance and store
            columns_with_importance = feature_importance(columns_with_indices, model, X_train, num_repeats, columns_with_importance)
        total_acc = np.asarray(total_acc)
        print("Average Acc:", np.mean(total_acc), np.var(total_acc))
        #########################################################################################################
        
        
        
        
        
        ####################################### TEST DATA PREPROCESSING ############################################
        # Now, preprocess the test data
        if target_type == "dnb1":
            test_dir = "DnB1/"+chunk+"/test/"
        else:
            test_dir = "DnB1+DnB2/"+chunk+"/test/"
        file_names = [file_name for file_name in os.listdir(test_dir) if file_name.endswith('.xlsx')]
        X_test_seq, X_test_static, Y_test = list(), list(), list()
        for file_name in tqdm(file_names):
            df = pd.DataFrame(pd.read_excel(test_dir+file_name, engine="calamine"))
            # Get all the static values
            static_feats = np.concatenate((df[["baseline_pd", "baseline_gsr"]].values[0], df[["gsr_responses"]].mean()))
            # static_feats = df[["baseline_pd"]].values[0]
            X_test_static.append(static_feats)
            # Drop extra columns
            # df = df.drop(["T2LF", "T2FF", "TSD", "total_responses", "TFD", "NOF", "reaction_time (s)", "baseline_gsr", "baseline_pd", "gsr_responses"], axis=1)
            df = df.drop(["T2LF", "T2FF", "TSD", "total_responses", "TFD", "NOF", "reaction_time (s)", "baseline_gsr", "baseline_pd", "gsr_responses", "avg_gsr_amplitude", "gsr_response_proportion"], axis=1)
            # Impute missing values
            if df.isnull().values.any():
                curr_X = df.values
                # imp_mean = IterativeImputer(RandomForestRegressor(random_state=27), random_state=27)
                imp_mean = IterativeImputer(random_state=27)
                imp_mean.fit(curr_X)
                curr_X = imp_mean.transform(curr_X)
            else:
                curr_X = df.values
            scaler = StandardScaler()
            curr_X = scaler.fit_transform(curr_X)
            # Get the data in correct format for machine learning
            curr_X = curr_X.flatten()
            X_test_seq.append(curr_X)
            curr_Y = target[file_name[:file_name.index(".")]]
            Y_test.append(curr_Y)
        X_test_seq, X_test_static, Y_test = np.asarray(X_test_seq), np.asarray(X_test_static), np.asarray(Y_test)
        
        X_test_static = scaler_static.transform(X_test_static)
        X_test = np.concatenate((X_test_seq, X_test_static), axis=1)
        #########################################################################################################
        
        
        
        
        
        ######################################### FINAL PREDICTION ###############################################
        # Finally, train the model on the entire data and test
        model = RandomForestClassifier(n_estimators=grid_search.best_params_["n_estimators"], 
                                        max_depth=grid_search.best_params_["max_depth"],
                                        min_samples_split=grid_search.best_params_["min_samples_split"],
                                        max_features=grid_search.best_params_["max_features"],
                                        random_state=27)
        model.fit(X_train, Y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(Y_test, preds)
        print("Accuracy on Test Set:", acc)
        #########################################################################################################
    
    
    
    
    
    ##################################### PLOTTING FEATURE IMPORTANCE #######################################
    names, vals = list(), list()
    for column_name, importance_scores in columns_with_importance.items():
        names.append(column_name)
        importance_scores = np.asarray(importance_scores)
        vals.append(np.mean(importance_scores))
    names, vals = np.asarray(names), np.asarray(vals)
    vals/= np.sum(vals)
    indices = np.argsort(vals)
    names, vals = names[indices], vals[indices]
    # Create a dataframe
    df = pd.DataFrame({
        "Feature": names,
        "Importance": vals
    })
    # Create horizontal bar plot
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Importance', y='Feature', data=df)
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=400)
    #########################################################################################################
    
    for i, j in zip(names, vals):
        print(i, j)