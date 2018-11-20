import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

def load_data(file,cols=None):
    df = pd.read_csv("{}".format(file),index_col="Date",parse_dates=True,usecols=cols,na_values=['nan'])
    return df
        
def merge_data(df1, df2):
    df = df1.join(df2,how='inner')
    #df = df_temp.join(df3, how='inner')
    return df

def load_dataset(file1,file2,file3,n=1):
    if 'nifty50' in file1:
        df1 = load_data(file1)
        df1 = df1.rename(columns={'Shares_Traded':'Volume'})
    if 'niftype' in file2:
        df2 = load_data(file2)
    if 'niftyvix' in file3:
        df3 = load_data(file3,['Date','Close'])
        df3 = df3.rename(columns={'Close':'Vix'})
    df4 = merge_data(df1,df2)
    df = merge_data(df4,df3)
    df['Close'] = df['Close'].shift(-n)
    df = df.dropna()
    prices = df['Close']
    features = df.drop(['Close'],axis=1)
    return features,prices
    #return df

def split_data(feature, label):
    X = feature
    y = label
    X_train, X_test, y_train, y_test = train_test_split(X,y,shuffle=False,test_size=0.2,random_state=42)
    return X_train,X_test,y_train,y_test

def minmax(data,feat):
    scaler = MinMaxScaler()
    data[feat] = scaler.fit_transform(data[feat])
    return data

def inverse_minmax(data,feat,scaler):
    data[feat] = scaler.inverse_transform(data[feat])
    return data

def normalize(data,row):
    data = data/row
    return data

def denormalize(data,row):
    data = data*row
    return data

def mean_normalize(data):
    data = data-np.mean(data)/np.std(data)
    return data

def train_predict(estimator, X, y,X_test,y_test):
    model = estimator.fit(X,y)
    y_pred = model.predict(X_test)
    acc = performance_metric(y_test,y_pred)
    return model, acc
    
def performance_metric(model,y_true,y_pred):
    acc = r2_score(y_true,y_pred)
    error = mean_squared_error(y_true,y_pred)
    print('Coefficients: \n', model.coef_)
    print("Mean Squared error: %.2f"% mean_squared_error(y_true,y_pred))
    print('Variance score: %.2f' % r2_score(y_true,y_pred))
    return acc, error


