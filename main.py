import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class LSTM_linear_(nn.Module):
    def __init__(self, inputs,outputs, features, hidden=256):
        super(LSTM_linear_, self).__init__()

        self.lstmlayer = nn.LSTM(input_size=features, hidden_size=hidden, num_layers=5, batch_first=True)
        self.linearlayer = nn.Linear(inputs*hidden, outputs*features)

    def forward(self, x):
        x_shape=x.shape
        x, _ = self.lstmlayer(x)
        x=x.reshape(x_shape[0],-1)
        x = self.linearlayer(x)

        x=x.reshape(x_shape[0],int(x.shape[1]/x_shape[2]),-1)
        return x


def load_data():
    # load data
    data = pd.read_csv('flow3.csv',header=None)
    data=data.loc[:,((data==-1).sum(axis=0)<1)]
    data.columns=[i for i in range(len(data.columns))]
    data=data.values
    data[data==1]=0
    '''
    data shape: [time,flow]

    '''
    return data.astype(np.float32)

    # data = pd.read_csv('flow3.csv',header=None)
    # data=data.values
    # # print(data[1]==-1)
    # data=data[:,data[0,:]>-1]

def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i in range(X.shape[0] - (num_timesteps_input + num_timesteps_output) + 1)]
    features, target = [], []
    for i, j in indices:
        features.append(X[i: i + num_timesteps_input,:])
        target.append(X[i + num_timesteps_input: j,:])
    a=np.array(features)
    return torch.from_numpy(a), \
           torch.from_numpy(np.array(target))


def train_epoch(training_input, training_target, batch_size):
    permutation = torch.randperm(training_input.shape[0])
    epoch_training_losses = []
    for i in range(0, training_input.shape[0], batch_size):
        net.train()
        # gradients set zero
        optimizer.zero_grad()
        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]
        X_batch = X_batch.to(device=device)
        y_batch = y_batch.to(device=device)
        out = net(X_batch)
        loss = loss_criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
    return sum(epoch_training_losses) / len(epoch_training_losses)


def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))


def smape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))) * 100


if __name__ == '__main__':
    torch.manual_seed(7)
    epochs = 1000
    batch_size = 256

    num_timesteps_input = 5
    num_timesteps_output = 5


    # 定义设备
    device = torch.device('cpu')

    # 加载数据
    X = load_data()
    links=X.shape[1]
    # 标准化
    means = np.mean(X)
    X = (X - means)
    stds = np.std(X)
    X = X / stds

    # 数据切分
    split_line0 = int(X.shape[0] * 0.8)
    split_line1 = int(X.shape[0] * 0.2)
    train_original_data = X[:split_line0,:]
    val_original_data = X[split_line0:,:]
    training_input, training_target = generate_dataset(train_original_data,
                                                       num_timesteps_input=num_timesteps_input,
                                                       num_timesteps_output=num_timesteps_output)


    val_input, val_target = generate_dataset(val_original_data,
                                             num_timesteps_input=num_timesteps_input,
                                             num_timesteps_output=num_timesteps_output)

    # 定义模型
    net = LSTM_linear_(num_timesteps_input,num_timesteps_output,links).to(device=device)
    # 定义优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    # 定义损失函数
    loss_criterion = nn.MSELoss()

    training_losses = []
    validation_losses = []
    validation_maes = []
    validation_mapes = []
    validation_smapes = []
    validation_rmses = []
    for epoch in range(epochs):
        loss = train_epoch(training_input, training_target,
                           batch_size=batch_size)

        training_losses.append(loss)

        with torch.no_grad():
            net.eval()
            val_input = val_input.to(device=device)
            val_target = val_target.to(device=device)

            out = net(val_input)
            val_loss = loss_criterion(out, val_target).to(device="cpu")
            validation_losses.append(np.asscalar(val_loss.detach().numpy()))

            # 反标准化
            out_unnormalized = out.detach().cpu().numpy() * stds+ means
            target_unnormalized = val_target.detach().cpu().numpy() * stds + means

            mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))
            rmse_ = rmse(target_unnormalized, out_unnormalized)
            mape_ = mape(target_unnormalized, out_unnormalized)
            smape_ = smape(target_unnormalized, out_unnormalized)
            validation_maes.append(mae)
            validation_rmses.append(rmse_)
            validation_smapes.append(smape_)
            # if (epoch+1)%300==0 and epoch!=0:
            #     plt.plot(target_unnormalized[0, 0, :], label="val")
            #     plt.plot(out_unnormalized[0, 0, :], label="out")
            #     plt.legend()
            #     plt.show()
            out = None
        print('epoch', epoch)
        print("Training loss: {}".format(training_losses[-1]))
        print("Validation loss: {}".format(validation_losses[-1]))
        print("means: ", means)
        print("Validation MAE: ", validation_maes[-1])
        print("Validation RMSE: ", validation_rmses[-1])
        print("Validation sMAPE: ", validation_smapes[-1])
    torch.save({'model':net}, "lstm.pth")
    plt.plot(training_losses)
    plt.legend()
    plt.show()


