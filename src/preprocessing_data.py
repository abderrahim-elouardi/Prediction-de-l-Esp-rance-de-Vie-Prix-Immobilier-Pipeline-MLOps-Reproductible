import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder , StandardScaler
import os



def drop_unuseful_columns(df):
    df_preprocessed = df.copy()
    if "Unnamed: 0" in df_preprocessed.columns:
        df_preprocessed.drop(columns=["Unnamed: 0"],inplace=True)
    if "Id" in df_preprocessed.columns:
        df_preprocessed.drop(columns=["Id"], inplace=True)

    return df_preprocessed

def drop_duplicates(df):
    df.drop_duplicates(inplace=True)
    return df

def process_messing_values(df):
    return df

def process_outliers(train_df_x,coefficient_de_clôture):
    index_to_drop = []
    for column in  train_df_x.select_dtypes(include=["int64"]).columns:
        Q1 = train_df_x[column].quantile(0.25)
        Q3 = train_df_x[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - coefficient_de_clôture * IQR
        upper_bound = Q3 + coefficient_de_clôture * IQR

        index_to_drop.extend(train_df_x[(train_df_x[column]<lower_bound) | (train_df_x[column]>upper_bound)].index)
    index_to_drop = list(set(index_to_drop))
    train_df_x.drop(index=index_to_drop , inplace=True)
    return train_df_x
    

def normalize(train_df_x , test_df_x):
    columns_to_norm = ["YearBuilt" , "Area"]
    scaler = StandardScaler()
    train_df_x[columns_to_norm] = scaler.fit_transform(train_df_x[columns_to_norm])
    test_df_x[columns_to_norm] = scaler.transform(test_df_x[columns_to_norm])
    return train_df_x , test_df_x

def encode(train_df_x , test_df_x):
    columns_to_encode = ["Location","Garage","Condition"]
    location_categories = ["Suburban", "Urban" , "Downtown" , "Rural"]
    garage_categories = ["Yes", "No"]
    condition_categories = ["Good", "Poor" , "Excellent" , "Fair"]

    oe = OrdinalEncoder(categories=[location_categories , garage_categories , condition_categories])
    train_df_x[columns_to_encode] = oe.fit_transform(train_df_x[columns_to_encode])
    test_df_x[columns_to_encode] = oe.transform(test_df_x[columns_to_encode])
    return train_df_x , test_df_x

def split_data(df_preprocessed , test_size=0.15,random_state=123 , shuffle=True):
    X = df_preprocessed.drop(columns=["Price"])
    y = df_preprocessed["Price"]
    train_df_x , test_df_x , train_df_y , test_df_y  = train_test_split(X , y , test_size=0.15,random_state=123 , shuffle=True)
    return train_df_x , test_df_x , train_df_y , test_df_y 

def preprocessing_dataset(source_data_path , destinaiton_dir_path , coefficient_de_clôture):
    df = pd.read_csv(source_data_path)
    # drop unuseful columns
    df_preprocessed = drop_unuseful_columns(df)

    # droping duplicates
    df_preprocessed = drop_duplicates(df_preprocessed)
    
    
    # splitting data
    X = df_preprocessed.drop(columns=["Price"])
    y = df_preprocessed["Price"]
    train_df_x , test_df_x , train_df_y , test_df_y  = split_data(X , y , test_size=0.15,random_state=123 , shuffle=True)


        
    # processing messing values
    df_preprocessed = process_messing_values(df_preprocessed)
    # processing outliers
    train_df_x = process_outliers(train_df_x , coefficient_de_clôture)

    

    # Nomalization
    train_df_x , test_df_x = normalize(train_df_x , test_df_x)
    
    # encoding
    train_df_x  , test_df_x = encode(train_df_x , test_df_x)
    train_df_x.to_csv(os.path.join(destinaiton_dir_path,"train_df_x.csv") , index=False)
    train_df_y.to_csv(os.path.join(destinaiton_dir_path,"train_df_y.csv") , index=False)
    test_df_x.to_csv(os.path.join(destinaiton_dir_path,"test_df_x.csv") , index=False)
    test_df_y.to_csv(os.path.join(destinaiton_dir_path,"test_df_y.csv") , index=False)



if __name__ == "__main__":
    preprocessing_dataset(
        "data/rows/01_dataset.csv",
        "data/processed/",
        1.5
    )