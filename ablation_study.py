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
from sklearn.preprocessing import StandardScaler
from PyImpetus import PPIMBC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from matplotlib.lines import Line2D
from copy import deepcopy
np.random.seed(26)





if __name__ == "__main__":
    ####################################### SET THIS PARAMETER #############################################
    target_type = "dnb1"#"dnb1+dnb2"#
    chunk = "chunks25"
    chunk_val = int(chunk[chunk.index("s")+1:])
    best_params_rf = {
        "n_estimators": 90,
        "max_depth": 3,
        "min_samples_split": 10,
        "max_features": "log2"
    }
    ablate_wm_group = "high"
    ########################################################################################################
    
    # First get the target
    target_dir = "target.xlsx"
    target_raw = pd.DataFrame(pd.read_excel(target_dir, engine="calamine"))
    # Convert to a dictionary
    target_raw = target_raw[["Name", "DnB1.1", "DnB1.2", "DnB2.1", "DnB2.2"]].values
    target = dict()
    target_val = list()
    # Decide the target based on the target type
    for name, score1, score2, score3, score4 in target_raw:
        if target_type == "dnb1":
            target[name.strip().lower()] = 1 if score1 >= 80 else 0
        else:
            target[name.strip().lower()] = 1 if (score1 >= 80 or score3 >= 80) else 0
        target_val.append(target[name.strip().lower()])
    # target_raw = pd.DataFrame(pd.read_excel(target_dir, engine="calamine"))
    # target_raw["Group"] = target_val
    # target_raw.to_excel("target_final_2.xlsx", index=False)
    # quit()
    

    if target_type == "dnb1":
            train_dir = "DnB1/"+chunk+"/train/"
    else:
        train_dir = "DnB1+DnB2/"+chunk+"/train/"
    file_names = [file_name for file_name in os.listdir(train_dir) if file_name.endswith('.xlsx')]
    file_names.remove("tnh_36.xlsx")#
    file_names.remove("tnh_38.xlsx")#
    # file_names.remove("tnh_67.xlsx")
    
    all_scalers = list()
    
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
        all_scalers.append(scaler)
        # Get the data in correct format for machine learning
        curr_X = curr_X.flatten()
        X_train_seq.append(curr_X)
        curr_Y = target[file_name[:file_name.index(".")]]
        Y_train.append(curr_Y)
    all_scalers = np.asarray(all_scalers)
    X_train_seq, X_train_static, Y_train = np.asarray(X_train_seq), np.asarray(X_train_static), np.asarray(Y_train)
    # Standardize the static features and concatenate with sequential features
    scaler_static = StandardScaler()
    X_train_static = scaler_static.fit_transform(X_train_static)
    X_train = np.concatenate((X_train_seq, X_train_static), axis=1)#
    
    model = RandomForestClassifier(n_estimators=best_params_rf["n_estimators"], 
                                    max_depth=best_params_rf["max_depth"],
                                    min_samples_split=best_params_rf["min_samples_split"],
                                    max_features=best_params_rf["max_features"],
                                    random_state=27)
    # model = LogisticRegression(random_state=27, C=best_params_lr["C"], penalty=best_params_lr["penalty"], solver="liblinear")
    model.fit(X_train, Y_train)
    preds = model.predict_proba(X_train)
    
    
    num_people = 34
    
    # Randomly pick num_people from low working memory
    low_wm_indices = np.where(Y_train == 0)[0]
    np.random.shuffle(low_wm_indices)
    low_wm_indices = low_wm_indices[:num_people]
    # Randomly pick num_people from high working memory
    high_wm_indices = np.where(Y_train == 1)[0]
    np.random.shuffle(high_wm_indices)
    high_wm_indices = high_wm_indices[:num_people]
    
    if ablate_wm_group == "high":
        final_indices = high_wm_indices
    else:
        final_indices = low_wm_indices
    
    features = ["AFD", "PV", "MSV", "mean_gsr_responses", "MPD", "MSA"]
    # features = ["AFD"]
    # For each feature find its corresponding mean value in high working memory
    mean_value = dict()
    final_mean, final_std = 0, 0
    for feature in features:
        if ablate_wm_group == "high":
            rows = X_train[np.where(Y_train==0)[0]]
            # First transform the values back to original input
            scalers = all_scalers[np.where(Y_train==0)[0]]
            # We need to do this separately for each person
            for person_idx in range(len(rows)):
                if feature!= "mean_gsr_responses":
                    mean = scalers[person_idx].mean_[columns_with_indices[feature][0]]
                    std = np.sqrt(scalers[person_idx].var_[columns_with_indices[feature][0]])
                    final_mean+= mean
                    final_std+= std
                    rows[person_idx, columns_with_indices[feature]] = rows[person_idx, columns_with_indices[feature]] * std + mean
                else:
                    # Static variables have only one scaler
                    mean = scaler_static.mean_[-1]
                    std = np.sqrt(scaler_static.var_[-1])
                    final_mean+= mean
                    final_std+= std
                    rows[person_idx, columns_with_indices[feature]] = rows[person_idx, columns_with_indices[feature]] * std + mean
            mean_value[feature] = np.mean(rows[:, columns_with_indices[feature]])
            # new_rows = X_train[np.where(Y_train==1)[0]]
            # mean_val = np.mean(new_rows[:, columns_with_indices[feature]])
            # mean_value[feature]+= (mean_value[feature] - mean_val)
        else:
            rows = X_train[np.where(Y_train==1)[0]]
            # First transform the values back to original input
            scalers = all_scalers[np.where(Y_train==1)[0]]
            # We need to do this separately for each person
            for person_idx in range(len(rows)):
                if feature!= "mean_gsr_responses":
                    mean = scalers[person_idx].mean_[columns_with_indices[feature][0]]
                    std = np.sqrt(scalers[person_idx].var_[columns_with_indices[feature][0]])
                    final_mean+= mean
                    final_std+= std
                    rows[person_idx, columns_with_indices[feature]] = rows[person_idx, columns_with_indices[feature]] * std + mean
                else:
                    # Static variables have only one scaler
                    mean = scaler_static.mean_[-1]
                    std = np.sqrt(scaler_static.var_[-1])
                    final_mean+= mean
                    final_std+= std
                    rows[person_idx, columns_with_indices[feature]] = rows[person_idx, columns_with_indices[feature]] * std + mean
            mean_value[feature] = np.mean(rows[:, columns_with_indices[feature]])
            # new_rows = X_train[np.where(Y_train==0)[0]]
            # mean_val = np.mean(new_rows[:, columns_with_indices[feature]])
            # mean_value[feature]+= (mean_value[feature] - mean_val)
        final_mean/= len(rows)
        final_std/= len(rows)
        print(mean_value[feature])
        mean_value[feature] = (mean_value[feature]-final_mean)/final_std
        print(mean_value[feature])
    
    # We now ablate feature
    feature_change = [list() for i in range(num_people)]
    lr = 0.1
    num_epochs = 100
    orig_copy = deepcopy(X_train[final_indices])
    for i in tqdm(range(num_epochs)):
        print("\nMovement index:", i+1, "\n---------------\n")
        for feature in features:
            rows = X_train[final_indices]
            rows[:, columns_with_indices[feature]]-= lr * (rows[:, columns_with_indices[feature]] - mean_value[feature])
            X_train[final_indices] = rows
            print(np.mean((rows[:, columns_with_indices[feature]]/orig_copy[:,columns_with_indices[feature]])))
        preds = model.predict_proba(X_train)
        for idx, index in enumerate(final_indices):
            feature_change[idx].append(preds[index,1])
        print("\n------------")
    
    for i in range(num_people):
        print(feature_change[i])
        sns.lineplot(x=np.arange(num_epochs), y=feature_change[i])#, label="particiapnt "+str(i+1))
        # plt.savefig(feature+"_ablation_study.png", dpi=400)
    sns.lineplot(x=np.arange(num_epochs), y=np.ones(num_epochs)*0.5, linestyle="--", color="black")
    plt.xlabel("Number of increments")
    plt.ylabel("Probability")
    plt.ylim(0.0, 1)
    # Create a proxy line for legend
    participant_line = Line2D([0], [0], color='gray', label='Participants')
    threshold_proxy = Line2D([0], [0], color='black', linestyle='--', label='Classification threshold (p = 0.5)')
    # Add the legend
    plt.legend(handles=[participant_line, threshold_proxy])
    plt.tight_layout()
    plt.savefig("Top"+str(len(features))+"_ablation_study_"+ablate_wm_group+"_WM.png", dpi=400)