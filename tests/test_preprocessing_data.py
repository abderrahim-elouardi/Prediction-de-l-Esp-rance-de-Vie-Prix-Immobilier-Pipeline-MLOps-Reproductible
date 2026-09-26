from src.preprocessing_data import  encode , process_outliers  ,drop_duplicates,normalize,drop_unuseful_columns , process_messing_values , split_data
import pandas as pd
import pytest



def test_drop_unuseful_columns():
    #arrange
    df = pd.read_csv("data/rows/01_dataset.csv")
    df = df.sample(n=100, random_state=42)
    #action
    df = drop_unuseful_columns(df)

    #  
    assert "Id" not in df.columns
    assert "Unnamed: 0" not in df.columns
        

def test_drop_duplicates():
    #arrange
    df = pd.read_csv("data/rows/01_dataset.csv")
    df = df.sample(n=100, random_state=42)
    #action
    df = drop_duplicates(df)
    assert df.duplicated().sum()==0


# def encode_test():



def test_process_outliers():
    #arrange
    df = pd.read_csv("data/rows/01_dataset.csv")
    df = df.sample(n=100, random_state=42)
    #action
    df = process_outliers(df,1.5)
    index_to_drop = []
    for column in  df.select_dtypes(include=["int64"]).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        index_to_drop.extend(df[(df[column]<lower_bound) | (df[column]>upper_bound)].index)
    index_to_drop = list(set(index_to_drop))

    assert len(index_to_drop) == 0

def test_normalize():
    #arrange
    df = pd.read_csv("data/rows/01_dataset.csv")
    df = df.sample(n=100, random_state=42)
    #action
    X_train , X_test , _ , _ = split_data(df)
    df = normalize(X_train , X_test)

    #check
    columns_to_norm = ["YearBuilt" , "Area"]
    for col in columns_to_norm:
        assert X_train[col].mean()==pytest.approx(0, abs=1e-5)
        assert X_train[col].var()== pytest.approx(1, abs=0.05)
                
            

def test_process_messing_values():
    #arrange
    df = pd.read_csv("data/rows/01_dataset.csv")
    df = df.sample(n=100, random_state=42)
    #action
    df = process_messing_values(df)

    assert df.isna().sum().sum() == 0
