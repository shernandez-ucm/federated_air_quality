import pandas as pd
import jax
import jax.numpy as jnp

def pre_process(df):
    feature_keys = ['pm25','pm10','no2','temp','velv','hrel']
    df.columns = df.columns.str.strip()
    df['date_time']=pd.to_datetime(df[['anho', 'mes', 'dia', 'hora']].rename(columns={'anho': 'year', 'mes': 'month', 'dia': 'day', 'hora': 'hour'}))
    df.drop(columns=['anho', 'mes', 'dia','hora'],inplace=True)
    df.drop(columns=['numero_semana', 'numero_dia', 'tipo_dia','estacion_anho'],inplace=True)
    df.set_index('date_time',inplace=True)
    for key in feature_keys:
        if key in df.columns:
            df[key] =df[key].astype(str).str.replace(',', '.').astype(float)
            #df[key].fillna(df[key].mean(), inplace=True)  
    df.resample('1h').sum()      
    return df

def train_test_split(data,split_fraction,feature_keys):
    data=data[feature_keys]
    train_split = int(split_fraction * int(data.shape[0]))
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    data = (data - data_mean) / data_std
    train_data = data.iloc[0 : train_split - 1]
    val_data = data.iloc[train_split:]
    return train_data,val_data

def create_batch(data,lag,future):
    df_lag=pd.concat([data[:-future].shift(i) for i in range(lag-1,-1,-1)],axis=1).dropna()
    #df_lag.columns=['pm_'+str(i) for i in range(lag,-1,1)]
    X=df_lag.values
    y=data[future+lag-1:].values
    return X,y

def create_batch_multistep(df,lag,future,feature=None):
    if feature is None:
        data=df
    else:
        data=df[feature]
    df_lag=pd.concat([data[:-future].shift(i) for i in range(lag-1,-1,-1)],axis=1).dropna()
    df_future=pd.concat([data[lag-1:].shift(-i) for i in range(1,future+1)],axis=1).dropna()
    #df_lag.columns=['pm_'+str(i) for i in range(lag,-1,1)]
    X=df_lag.values
    y=df_future.values
    return X,y

def get_dataloader(X,y,batch_size,key,axis=0):
    num_train=X.shape[axis]
    indices = jnp.array(list(range(0,num_train)))
    indices=jax.random.permutation(key,indices)
    for i in range(0, len(indices),batch_size):
        batch_indices = jnp.array(indices[i: i+batch_size])
        yield X[:,batch_indices,:,:], y[:,batch_indices]