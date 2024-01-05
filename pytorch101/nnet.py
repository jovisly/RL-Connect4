import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim

from utils import get_data


HL_SIZE = 10
NUM_EPOCH = 320
BATCH_SIZE = 10

class Classifier(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.hidden1 = nn.Linear(num_features, HL_SIZE)
        self.activation1 = nn.ReLU()
        self.hidden2 = nn.Linear(HL_SIZE, num_features)
        self.activation2 = nn.ReLU()
        self.output = nn.Linear(num_features, 1)
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        x = self.activation1(self.hidden1(x))
        x = self.activation2(self.hidden2(x))
        x = self.act_output(self.output(x))
        return x

X_train, X_test, y_train, y_test = get_data()

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

num_features = X_train.shape[1]
nn_model = Classifier(num_features)

# Loss function and optimizer
loss_fn = nn.BCELoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.001)

for epoch in range(NUM_EPOCH):
    total_loss = 0
    num_batches = 0
    for i in range(0, len(X_train), BATCH_SIZE):
        num_batches += 1
        Xbatch = X_train[i:i+BATCH_SIZE]
        y_pred = nn_model(Xbatch)
        ybatch = y_train[i:i+BATCH_SIZE]
        loss = loss_fn(y_pred, ybatch)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 100 == 0:
      print(f'Epoch {epoch}; average loss {round(total_loss / num_batches, 6)}')



with torch.no_grad():
    y_pred = nn_model(X_test)

auc = roc_auc_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_pred)

# Plot ROC curve
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: NNet')
plt.legend(loc='lower right')
plt.savefig('nnet.png')

print(f'Model AUC: {auc:.2f}')
