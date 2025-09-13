import sys
import torch
from torch import nn
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("./data/SMSSpamCollection", 
    sep="\t", 
    names=["type", "message"])

df["spam"] = df["type"] == "spam"
df.drop("type", axis=1, inplace=True)

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

def evaluate_model(X, Y):
    model.eval()
    with torch.no_grad():
        y_pred = nn.functional.sigmoid(model(X)) > 0.25
        print("Predicted output: ", y_pred)

        # Accuracy -> fraction of correct predictions(spam or not spam)
        accuracy = (y_pred == Y).type(torch.float32).mean()
        print("Accuracy:", accuracy)
        # Sensitivity -> fraction of actual spam that are correctly predicted as spam
        sensitivity = (y_pred[Y == 1] == Y[Y == 1])\
            .type(torch.float32).mean()
        print("Sensitivity:", sensitivity)
        # Specificity -> fraction of actual not spam that are correctly predicted as not spam
        specificity = (y_pred[Y == 0] == Y[Y == 0])\
            .type(torch.float32).mean()
        print("Specificity:", specificity)
        # Precision -> fraction of predicted spam, that are actually spam
        precision = (y_pred[y_pred == 1] == Y[y_pred == 1])\
            .type(torch.float32).mean()
        print("Precision:", precision)

print("Evaluating on the training data")
evaluate_model(X_train, Y_train)

print("Evaluating on the validation data")
evaluate_model(X_val, Y_val)

# ----
# Testing the model
custom_messages = cv.transform([
    "We have release a new product, do you want to buy it?", 
    "Winner winner chicken dinner!",
    "Wake up, its the first day of the month!"
])

X_custom = torch.tensor(custom_messages.todense(), dtype=torch.float32)

model.eval()
with torch.no_grad():
    prediction = nn.functional.sigmoid(model(X_custom)) 
    print(prediction)