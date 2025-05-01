import torch
import torch.nn
from dataset import DataSetObject, Big2Dataset
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
import random



class NeuralNetwork(torch.nn.Module):
  def __init__(self, input_nodes=56, output_nodes=4):
        super(NeuralNetwork, self).__init__()
        #self.fc1 = torch.nn.Linear(input_nodes, 256)
        self.fc2 = torch.nn.Linear(input_nodes, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, output_nodes)
        self.act_fn = torch.nn.ReLU()

  def forward(self, X):
    #X = self.act_fn(self.fc1(X))
    X = self.act_fn(self.fc2(X))
    X = self.act_fn(self.fc3(X))
    X = self.fc4(X)
    return X
  

def train(model, dataloader, optimizer, n_epochs, alpha, alpha_decay, alpha_decay_period, device = "cuda"):
  device = torch.device(device)
  model.to(device)
  print(device)

  loss_fn = torch.nn.CrossEntropyLoss()
  
  for epoch in range(n_epochs):
    n_batches = 0
    if epoch % alpha_decay_period and epoch != 0:
      for param_group in optimizer.param_groups:
            param_group["lr"] *= alpha_decay

    tot_loss = 0
    for idx, (x, y) in enumerate(dataloader):
      n_batches +=1
      x = x.to(device)
      y = y.to(device)
      y_hat = model(x)
      loss = loss_fn(y_hat, y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      tot_loss += loss
    #print(f"Epoch: {epoch + 1}, avg loss: {tot_loss/n_batches}")

  return model

def eval(model, dataloader, device="cuda"):
  device = torch.device(device)
  model.to(device)
  n_correct = 0
  n_sample = 0

  with torch.no_grad():
        for (x, y) in dataloader:
            # TODO: Move images and labels to device using .to()
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)

            y_hat = y_hat.argmax(1)

            # Accumulate number of samples
            n_sample += y_hat.shape[0]

            # TODO: Check if our prediction is correct
            n_correct  += torch.sum(y_hat == y)
  accuracy = n_correct/ n_sample
  print(f"Test Size: {n_sample}, accuracy: {accuracy}")
  return accuracy


def test_knn(data, data_test, k, p = 2):
  model = KNeighborsClassifier(k,p=p)
  model.fit(data.x, data.y)
  y_hat = model.predict(data_test.x)

  n_samples = len(y_hat)

  n_correct = sum(y_hat == data_test.y)
  print(n_correct, n_samples, n_correct/ n_samples)
  return n_correct/ n_samples

        
if __name__ == "__main__":
    random.seed(42)

    #create training data
    data = DataSetObject(1000000)
    dataset = Big2Dataset(data.x, data.y)
    dataloader = DataLoader(dataset=dataset,  batch_size = 64)

    #create testing data
    data_test = DataSetObject(10000)
    dataset_test = Big2Dataset(data_test.x, data_test.y)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size = 64)

    #create validation data
    data_val = DataSetObject(10000)
    dataset_val = Big2Dataset(data_val.x, data_val.y)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size = 64)

    #initialize and use MLP for prediction
    model = NeuralNetwork()
    model = model.to("cuda")
    n_epochs  = 20
    alpha = 0.001
    alpha_decay = 0.95
    alpha_decay_period = 2

    optimizer = torch.optim.Adam(model.parameters(), lr = alpha)

    num_epochs = [5, 10, 20, 30]
    accuracies = []
    for num_epoch in num_epochs:
      model = NeuralNetwork().to("cuda")
      optimizer = torch.optim.Adam(model.parameters(), lr = alpha)
      model.train()
      model = train(model, dataloader, optimizer, num_epoch, alpha, alpha_decay, alpha_decay_period)
      model.eval()
      accuracies.append(eval(model, dataloader_test))
    print(accuracies)


    #initialize and use RNN for predictions
    print("For Minkowski distance")
    ks = [10, 30, 100, 300, 1000, 3000, 10000]
    accuracies = []
    for k in ks:
      accuracies.append(test_knn(data, data_test, k, p=2))

    print(accuracies)
    print(test_knn(data, data_val, 100,2))
    print(test_knn(data, data_val, 300,2))

    print("For euclidian Distance")
    accuracies = []

    for k in ks:
      accuracies.append(test_knn(data, data_test, k, p=1))

    print(accuracies)
    print(test_knn(data, data_val, 100,1))
    print(test_knn(data, data_val, 300,1))