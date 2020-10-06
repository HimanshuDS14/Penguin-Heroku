import pandas as pd
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("Penguins_data.csv")

df = data.copy()


target = ['species']
encode = ['sex' , 'island']


for col in encode:
    dummy = pd.get_dummies(df[col] , prefix = col)
    df = pd.concat([df , dummy] , axis =1)
    del df[col]
    


target_mapper = {'Adelie' : 0 , 'Chinstrap' : 1 , 'Gentoo' : 2}

def target_encode(val):
    return target_mapper[val]


df['species'] = df['species'].apply(target_encode)


x = df.drop('species' , axis = 1)
y = df['species']

model = RandomForestClassifier()
model.fit(x,y)


import pickle

pickle.dump(model , open("Penguin_model.pkl" , 'wb'))

