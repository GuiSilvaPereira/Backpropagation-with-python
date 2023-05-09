#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

SEED=10
np.random.seed(SEED)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mean_squared_error(predictions, labels):
    N = labels.size
    mse = ((predictions - labels)**2).sum() / (2*N)
    
    return mse

def accuracy(predictions, labels):
    predicions_correct = predictions.argmax(axis=1) == labels.argmax(axis=1)
    accuracy = predicions_correct.mean()
    
    return accuracy


# In[ ]:


def BPcreate(V, TrainSet, learning_rate = 0.1, epochs = 10000):
    SEED=10 
    y_train = pd.get_dummies(np.array(TrainSet.iloc[:,-1])).values
    x_train = TrainSet.iloc[:,:-1]

    N = y_train.size
    n_input = x_train.shape[1]
    n_hidden = V
    n_output = y_train.shape[1]

    np.random.seed(SEED)
    weights_1 = np.random.normal(scale=0.5, size=(n_input, n_hidden))   
    weights_2 = np.random.normal(scale=0.5, size=(n_hidden, n_output)) 

    # training the neural net
    monitoring = {"mean_squared_error": [], "accuracy": []}
    for epoch in range(epochs):    

        # feedforward
        hidden_layer_inputs = np.dot(x_train, weights_1)
        hidden_layer_outputs = sigmoid(hidden_layer_inputs)

        output_layer_inputs = np.dot(hidden_layer_outputs, weights_2)
        output_layer_outputs = sigmoid(output_layer_inputs)
        
        # monitor training process
        mse = mean_squared_error(output_layer_outputs, y_train)
        acc = accuracy(output_layer_outputs, y_train)

        monitoring["mean_squared_error"].append(mse)
        monitoring["accuracy"].append(acc)
    

      # backpropagation
        output_layer_error = output_layer_outputs - y_train
        output_layer_delta = output_layer_error * output_layer_outputs * (1 - output_layer_outputs)

        hidden_layer_error = np.dot(output_layer_delta, weights_2.T)
        hidden_layer_delta = hidden_layer_error * hidden_layer_outputs * (1 - hidden_layer_outputs)


        # weight updates
        weights_2_update = np.dot(hidden_layer_outputs.T, output_layer_delta) / N
        weights_1_update = np.dot(x_train.T, hidden_layer_delta) / N

        weights_2 = weights_2 - learning_rate * weights_2_update
        weights_1 = weights_1 - learning_rate * weights_1_update
        
    monitoring_df = pd.DataFrame(monitoring)
    fig, axes = plt.subplots(1, 2, figsize=(15,5))
    monitoring_df.mean_squared_error.plot(ax=axes[0], title="Mean Squared Error in TrainSet")
    monitoring_df.accuracy.plot(ax=axes[1], title="Accuracy in TrainSet")
    return list([weights_1, weights_2])


# In[ ]:


def BPpredict(NN, TestSet):
    # feedforward
    hidden_layer_inputs = np.dot(TestSet[:], NN[0])
    hidden_layer_outputs = sigmoid(hidden_layer_inputs)

    output_layer_inputs = np.dot(hidden_layer_outputs, NN[1])
    output_layer_outputs = sigmoid(output_layer_inputs)
    
    return output_layer_outputs


# # Iris dataset
# Data mining course

# In[ ]:


from sklearn.model_selection import train_test_split

url="Iris.csv"
df= pd.read_csv(url,sep=';',decimal=',',index_col=0)
df_train, df_test = train_test_split(df, test_size=0.2,stratify=df.loc[:,'species'])

# With stratify we guarantee that test and train will have the same % of each class ('species')

y_test = pd.get_dummies(np.array(df_test.loc[:,'species'])).values # 1st this because below we drop this column
df_test = df_test.drop(["species"], axis=1).values


# In[ ]:


V=50
NN=BPcreate(V,df_train,learning_rate = 0.015, epochs = 10000)
results=BPpredict(NN=NN,TestSet=df_test)

print(f'The accuracy on TestSet is: %.2f%%'%(accuracy(results,y_test)*100))


# # Wine Dataset
# Data mining course

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


url_wine="wine_dataset.csv"
df_wine= pd.read_csv(url_wine,sep=';',decimal='.',index_col=None)
df_train_w, df_test_w = train_test_split(df_wine, test_size=0.2,stratify=df_wine.iloc[:,-1])
y_test_w = pd.get_dummies(np.array(df_test_w.iloc[:,-1])).values
df_test_w = pd.DataFrame(data =df_test_w.drop(["CLASS"], axis=1).values, columns=df_train_w.iloc[:,:-1].keys())

#Scaler
scaler1 = StandardScaler() 
scaler1.fit(df_train_w.iloc[:,:-1])
x_v1 = scaler1.transform(df_test_w)  
df_train_w.iloc[:,:-1]= scaler1.transform(df_train_w.iloc[:,:-1])

df_test_w = pd.DataFrame(data = x_v1, columns =df_wine.iloc[:,:-1].keys() )
df_train_w = pd.DataFrame(data = df_train_w, columns =df_wine.keys() )


# In[ ]:


V = 35
NN=BPcreate(V,df_train_w,learning_rate = 0.1,epochs = 1000)
results=BPpredict(NN=NN,TestSet=df_test_w)

print(f'The accuracy on TestSet is: %.2f%%'%(accuracy(results,y_test_w)*100))


# # Data Mining I Dataset
# Data mining course

# In[ ]:


url_train= "training.teste1.csv"
url_test ="task.teste1.csv"
url_y_test="Best.csv" # Predictions for Kaggle

data_train=pd.read_csv(url_train, sep=";").iloc[:,1:]
data_test=pd.read_csv(url_test, sep=";").iloc[:,2:]

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

#Missing values
imp = KNNImputer(missing_values=np.nan,weights='distance',n_neighbors=6)
imp.fit(data_test)
data_test_dm=pd.DataFrame(data =imp.transform(data_test), columns=data_train.iloc[:,:-1].keys())
data_train.iloc[:,:-1]=imp.transform(data_train.iloc[:,:-1])
#Scaler
scaler1 = StandardScaler()
scaler1.fit(data_train.iloc[:,:-1])
x_v1 = scaler1.transform(data_test_dm) 
data_train.iloc[:,:-1]= scaler1.transform(data_train.iloc[:,:-1])

data_test = pd.DataFrame(data = x_v1, columns =data_test.keys() )
data_train = pd.DataFrame(data = data_train, columns =data_train.keys() )

y_test=pd.read_csv(url_y_test, sep=",").iloc[:,1:]
y_test_dm = pd.get_dummies(np.array(y_test.iloc[:,-1])).values


# In[ ]:


V = 150
NN=BPcreate(V,data_train,learning_rate =0.3, epochs = 1000)
results=BPpredict(NN=NN,TestSet=data_test)

print(f'The accuracy on TestSet is: %.2f%%'%(accuracy(results,y_test_dm)*100))


# Looking for class predictions with value 1, because without the data preprocessing or few layers, the predictions are always 0.

# In[ ]:


L=[]
for i in range(len(results)):
    if results[i][1]>results[i][0]:
        L.append(i)
# L
len(L)


# In[ ]:


V = 15
NN=BPcreate(V,data_train,learning_rate =0.15, epochs = 1000)
results_15=BPpredict(NN=NN,TestSet=data_test)

print(f'The accuracy on TestSet is: %.2f%%'%(accuracy(results_15,y_test_dm)*100))


# In[ ]:


L_15=[]
for i in range(len(results_15)):
    if results_15[i][1]>results_15[i][0]:
        L_15.append(i)
# L
len(L_15)

