import sys
import os
import torch
from torch import nn
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle

df = pd.read_csv("./data/SMSSpamCollection", 
    sep="\t", 
    names=["type", "message"])

df["spam"] = df["type"] == "spam"
df.drop("type", axis=1, inplace=True)

if not os.path.isdir("./model"):
    os.mkdir("./model")

# Training data
df_train = df.sample(frac=0.8, random_state=0)
# Validation data
df_val = df.drop(index=df_train.index)

cv = CountVectorizer(max_features=5000)
messages_train = cv.fit_transform(df_train["message"])
messages_val = cv.transform(df_val["message"])

X_train = torch.tensor(messages_train.todense(), dtype=torch.float32)
Y_train = torch.tensor(df_train["spam"].values, dtype=torch.float32)\
        .reshape((-1, 1))

X_val = torch.tensor(messages_val.todense(), dtype=torch.float32)
Y_val = torch.tensor(df_val["spam"].values, dtype=torch.float32)\
        .reshape((-1, 1))

model = nn.Linear(5000, 1)
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

for i in range(0, 50000):
    # Training pass
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = loss_fn(outputs, Y_train)
    loss.backward()
    optimizer.step()

    if i % 5000 == 0: 
        print(loss)

torch.save(model.state_dict(), "./model/model.pt")

with open('./model/vectorizer.pkl', 'wb') as f:
    pickle.dump(cv, f)

print("Vectorizer saved to ./model/vectorizer.pkl")




    