import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size * 3, hidden_size, num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # # Initialize hidden state with zeros
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # # Initialize cell state with zeros
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # # Forward propagate LSTM
        # out, _ = self.lstm(x, (h0, c0))
        
        # # Decode the hidden state of the last time step
        # out = self.fc(out[:, -1, :])

        B = x.size(0)
        x = x.view(B, 32, 32 * 3)

        x, (h_n, c_n) = self.lstm(x)

        x = x[:, -1, :]  # need check

        out = self.fc(x)

        return out


if __name__ == "__main__":
    input_size = 32  # 輸入特徵的維度
    hidden_size = 128  # LSTM 隱藏層的大小
    num_layers = 5  # LSTM 層的數量
    output_size = 10  # 輸出的大小，這裡假設是一個連續值的預測任務

    model = LSTMModel(input_size, hidden_size, num_layers, output_size)

    fake = torch.rand(13, 3, 32, 32)

    out = model(fake)

    print(out.shape)
