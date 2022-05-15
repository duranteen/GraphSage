import torch
from torch import nn
from net import GraphSage
from CoraData import CoraData
from sampling import multi_hop_sampling
import numpy as np

input_dim = 1433
hidden_dim = [128, 7]
num_neighbor_list = [10, 10]
batch_size = 32
num_epochs = 20
num_batch_pre_epoch = 20
learning_rate = 0.01
weight_decay = 5e-4

device = "cuda" if torch.cuda.is_available() else "cpu"

data = CoraData(rebuild=True).data
x = data.x / data.x.sum(1, keepdims=True)
train_index = np.where(data.train_mask)[0]
train_label = data.y
test_index = np.where(data.test_mask)[0]

model = GraphSage(input_dim, hidden_dim, num_neighbor_list).to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def train():
    model.train()
    for epoch in range(num_epochs):
        for batch in range(num_batch_pre_epoch):
            batch_src_index = np.random.choice(train_index, size=(batch_size,))
            batch_src_label = torch.from_numpy(train_label[batch_src_index]).long().to(device)
            batch_sampling_result = multi_hop_sampling(batch_src_index, num_neighbor_list, data.adjacency)
            batch_sampling_x = [torch.from_numpy(x[idx]).float().to(device) for idx in batch_sampling_result]
            batch_train_logits = model(batch_sampling_x)
            loss = criterion(batch_train_logits, batch_src_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Epoch {:03d} Batch {:03d} Loss: {:.4f}".format(epoch, batch, loss.item()))
        test()


def test():
    model.eval()
    with torch.no_grad():
        test_sampling_result = multi_hop_sampling(test_index, num_neighbor_list, data.adjacency)
        test_x = [torch.from_numpy(x[idx]).float().to(device) for idx in test_sampling_result]
        test_logits = model(test_x)
        test_label = torch.from_numpy(data.y[test_index]).long().to(device)
        predict_y = test_logits.max(1)[1]
        accuarcy = torch.eq(predict_y, test_label).float().mean().item()
        print("Test Accuracy: ", accuarcy)


train()