import pandas as pd
import numpy as np
from sklearn import preprocessing, datasets


def get_wine():
    data = datasets.load_wine()
    df = pd.DataFrame(data['data'])
    df['class'] = data['target']
    return df

def get_df(path, sep=',',columns=None):
    df = pd.read_csv(path,sep=sep,index_col=0)
    if columns is not None:
        df.columns = columns
    return df
    
def clean_df(df,label_columns):
    # onehot encode categorical variables
    y = df[label_columns]
    df = pd.get_dummies(df.drop(label_columns,axis=1),prefix_sep='_',drop_first=False)
    
    # Min max normalize everything
    mms = preprocessing.MinMaxScaler()
    df[df.columns] = mms.fit_transform(df[df.columns])
       
    # label encode label
    le = preprocessing.LabelEncoder()
    df[label_columns] = le.fit_transform(y)
    
    return df

def save_df(df,path):
    df.to_csv(path)
    
    
def split_df(df,label,ratio = .2):
    
    labels = df[label].unique()
    train = pd.DataFrame(columns=df.columns)
    test = pd.DataFrame(columns=df.columns)
    for l in labels:
        mask = df[label] == l
        split = int(np.floor(df[mask].shape[0] * 0.2))
        train = train.append(df[mask].iloc[:-split],ignore_index=True)
        test = test.append(df[mask].iloc[-split:],ignore_index=True)
    
    print(train.shape,test.shape)    
    return train,test
    

    
def run(dataset='wage'):
    if dataset == 'wage':
        df = get_df('balanced_wage.csv')
        print(df.columns)
        df = clean_df(df,'wage-class')
        print(df.head())
        train,test = split_df(df,'wage-class')
        save_df(train,'balanced_wage_cleaned_train.csv')
        save_df(test,'balanced_wage_cleaned_test.csv')
        print(train.groupby('wage-class').count(),test.groupby('wage-class').count())
        # save_df(df,'balanced_wage_cleaned.csv')
    elif dataset == 'wine':
        data = datasets.load_wine()
        df = pd.DataFrame(data['data'])
        df['class'] = data['target']
        # print(df.columns)
        df = clean_df(df,'class')
        # print(df.head())
        save_df(df,'balanced_wine_cleaned.csv')
        train,test = split_df(df,'class')
        save_df(train,'balanced_wine_cleaned_train.csv')
        save_df(test,'balanced_wine_cleaned_test.csv')
        print(train.groupby('class').count())
        
def run2(dataset='wage'):
    if dataset == 'wage':
        df = get_df('balanced_wage.csv')
        df = clean_df(df,'wage-class')
    elif dataset == 'wine':
        data = datasets.load_wine()
        df = pd.DataFrame(data['data'])
        df['class'] = data['target']
        df = clean_df(df,'class')
    
    
if __name__ == '__main__':
    run()