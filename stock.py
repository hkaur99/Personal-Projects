#!/usr/bin/env python
# coding: utf-8


# In[17]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pandas_datareader as pdr
import datetime as dt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from sklearn.preprocessing import MinMaxScaler


# In[18]:


company = 'SBIN.NS'


# In[19]:


start = dt.datetime(2011,1,1)

end = dt.datetime(2021,1,1)

data = pdr.DataReader(company, 'yahoo', start, end)


# In[20]:


data


# In[21]:


#data.to_csv('SBI.csv')


# In[22]:


#key  = '69238282d2eeb131e8e440407085a5e534ad523d'


# In[23]:


#data1 = pdr.get_data_tiingo('AAPL', api_key = key)


# In[24]:


#data1


# In[25]:


#data1.isnull().sum()


# In[26]:


#data1.to_csv('APPLE.csv')


# In[27]:


# Data preparation

#LSTM - Sequential 

# Train data - Accuracy

#past 100 days next 30 days next day ka stock value 

# 2nd algo


# In[28]:


#data1.tail()


# In[29]:


# overfit - train = 90%
            #test = 70% 
    
#underfit = train and test = 0


# In[30]:


data


# In[31]:


#Prepare data

scaler = MinMaxScaler(feature_range=(0,1))

scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
    


# In[32]:


scaled_data


# In[33]:


prediction_days = 60


# In[34]:


x_train = []
y_train = []


# In[35]:


for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x,0])


# In[36]:


x_train


# In[37]:


x_train, y_train = np.array(x_train), np.array(y_train)


# In[38]:


x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# In[39]:


model = Sequential()


# In[40]:


model.add(LSTM(units=60, return_sequences = True, input_shape=(x_train.shape[1],1)))


# In[41]:


model.add(Dropout(0.1))


# In[42]:


model.add(LSTM(units=60, return_sequences = True))


# In[43]:


model.add(Dropout(0.1))


# In[44]:


model.add(LSTM(units=60))


# In[45]:


model.add(Dense(units=1))


# In[46]:


model.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[47]:


model.fit(x_train, y_train, epochs = 70, batch_size = 32)


# In[48]:


#Load data

test_start = dt.datetime(2021,1,1)
test_end = dt.datetime(2021,8,10)


# In[49]:


test_data = pdr.DataReader(company, 'yahoo', test_start, test_end)


# In[50]:


test_data


# In[51]:


actual_price = test_data['Close'].values


# In[52]:


total_dataset = pd.concat((data["Close"], test_data['Close']))


# In[ ]:





# In[53]:


total_dataset


# In[54]:


model_input = total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values


# In[55]:


model_input


# In[56]:


model_input = model_input.reshape(-1,1)


# In[57]:


model_input = scaler.transform(model_input)


# In[58]:


x_test = []


# In[59]:


for x in range(prediction_days, len(model_input)):
    x_test.append(model_input[x-prediction_days:x,0])
    


# In[60]:


x_test = np.array(x_test)


# In[61]:


x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))


# In[62]:


predicted_price = model.predict(x_test)


# In[63]:


predicted_price = scaler.inverse_transform(predicted_price)


# In[64]:


#plot the days


# In[65]:


plt.plot(actual_price, color = 'r', label =f'Actual {company} price' )
plt.plot(predicted_price, color = 'g', label =f'Predicted {company} price' )
plt.title(f'{company} Share Price' )
plt.xlabel('Time')
plt.ylabel(f'{company} Share Price')
plt.legend()
plt.show()


# In[66]:


real_data = [model_input[len(model_input)+1 - prediction_days:len(model_input)+1,0]]


# In[67]:


real_data = np.array(real_data)


# In[68]:


real_data  = np.reshape(real_data, (real_data.shape[0],real_data.shape[1],1))


# In[69]:


prediction = model.predict(real_data)


# In[70]:


prediction = scaler.inverse_transform(prediction)


# In[71]:


print(f'Prediction: {prediction}')


# In[72]:


# 420/- +-


# In[73]:


import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model


# In[81]:


df= pd.read_csv("./stock_data.csv")

apps = dash.Dash()
server = apps.server

apps.layout = html.Div([
   
    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
   
    dcc.Tabs(id="tabs", children=[
       
        dcc.Tab(label='State Bank of India',children=[
html.Div([
html.H2("Actual price",style={"textAlign": "center"}),
dcc.Graph(
id="Actual Data",
figure={
"data":[
go.Scatter(
x=data.index,
y=actual_price,
mode='markers'
)

],
"layout":go.Layout(
title='scatter plot',
xaxis={'title':'Date'},
yaxis={'title':'Closing Rate'}
)
}

),
html.H2("LSTM Predicted closing price",style={"textAlign": "center"}),
dcc.Graph(
id="Predicteds Data",
figure={
"data":[
go.Scatter(
x=data.index,
y = predicted_price[:,0],
mode='markers'
)

],
"layout":go.Layout(
title='scatter plot',
xaxis={'title':'Date'},
yaxis={'title':'Closing Rate'}
)
}
)
])


        ]),
        dcc.Tab(label='Facebook Stock Data', children=[
            html.Div([
                html.H1("Stocks High vs Lows", 
                        style={'textAlign': 'center'}),
              
                dcc.Dropdown(id='my-dropdown',
                             options=[{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple','value': 'AAPL'}, 
                                      {'label': 'Facebook', 'value': 'FB'}, 
                                      {'label': 'Microsoft','value': 'MSFT'}], 
                             multi=True,value=['FB'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='highlow'),
                html.H1("Stocks Market Volume", style={'textAlign': 'center'}),
         
                dcc.Dropdown(id='my-dropdown2',
                             options=[{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple','value': 'AAPL'}, 
                                      {'label': 'Facebook', 'value': 'FB'},
                                      {'label': 'Microsoft','value': 'MSFT'}], 
                             multi=True,value=['FB'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='volume')
            ], className="container"),
        ])


    ])
])







@apps.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["High"],
                     mode='lines', opacity=0.7, 
                     name=f'High {dropdown[stock]}',textposition='bottom center'))
        trace2.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["Low"],
                     mode='lines', opacity=0.6,
                     name=f'Low {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Price (USD)"})}
    return figure


@apps.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["Volume"],
                     mode='lines', opacity=0.7,
                     name=f'Volume {dropdown[stock]}', textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data, 
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M',
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Transactions Volume"})}
    return figure



if __name__=='__main__':
    apps.run_server(debug=False)


# In[82]:


#get_ipython().run_line_magic('tb', '')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




