from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error
import pandas as pd 
import numpy as np
import mlflow
import joblib

df_train_x = pd.read_csv("data/processed/train_df_x.csv")
df_train_y = pd.read_csv("data/processed/train_df_y.csv")


df_train_x , df_val_x , df_train_y , df_val_y = train_test_split(df_train_x , df_train_y , random_state=123 , test_size=0.15)

# Doit afficher le chemin du package dans site-packages, PAS votre dossier local
print(mlflow.__file__)

mlflow.set_experiment(experiment_name="trainign_model_Esperance_de_vie")

with mlflow.start_run():
    exp_l1_ratio = 0.5
    exp_alpha = 1
    exp_num_ite = 1000
    model_lr = ElasticNet(alpha=exp_alpha , l1_ratio=exp_l1_ratio , fit_intercept=True , max_iter=exp_num_ite,random_state=123)
    model_lr.fit(df_train_x , df_train_y)

    train_predicted_y = model_lr.predict(df_train_x)
    val_predicted_y = model_lr.predict(df_val_x)

    R_train_mse = np.sqrt(mean_squared_error(df_train_y , train_predicted_y))
    R_val_mse = np.sqrt(mean_squared_error(df_val_y , val_predicted_y) )

    mlflow.log_param("exp_l1_ratio" , exp_l1_ratio)
    mlflow.log_param("exp_alpha" , exp_alpha)
    mlflow.log_param("exp_num_ite" , exp_num_ite)
    mlflow.sklearn.log_model(model_lr , "model_ElasticNet")
    mlflow.log_metric("rmse/num_obs" , R_val_mse/len(train_predicted_y))

    print(f"---- train mse : {R_train_mse/len(train_predicted_y)}")
    print(f"---- val mse : {R_val_mse/len(train_predicted_y)}")
    joblib.dump(model_lr, "models/mon_modele_elasticnet.joblib")

    