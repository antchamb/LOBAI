import torch
import torch.nn as nn


class Deeplob(nn.Module):
    def __init__(self, lighten):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.name = 'deeplob'
        if lighten:
            self.name += '-lighten'

        # convolution blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )

        if lighten:
            conv3_kernel_size = 5
        else:
            conv3_kernel_size = 10

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, conv3_kernel_size)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )

        # inception modules
        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )

        # lstm layers
        self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, 3)

    # def forward(self, x):
    #     batch_size = x.size(0)
    #     h0 = torch.zeros(1, batch_size, 64, device=x.device)
    #     c0 = torch.zeros(1, batch_size, 64, device=x.device)
    #
    #     x = self.conv1(x)
    #     x = self.conv2(x)
    #     x = self.conv3(x)
    #
    #     x_inp1 = self.inp1(x)
    #     x_inp2 = self.inp2(x)
    #     x_inp3 = self.inp3(x)
    #
    #     x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)
    #
    #     x = x.permute(0, 2, 1, 3)
    #     x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))
    #
    #     x, _ = self.lstm(x, (h0, c0))
    #     x = x[:, -1, :]
    #     x = self.fc1(x)
    #     forecast_y = torch.softmax(x, dim=1)
    #
    #     return forecast_y
    # def forward(self, x):
    #     batch_size = x.size(0)
    #     h0 = torch.zeros(1, batch_size, 64, device=x.device)
    #     c0 = torch.zeros(1, batch_size, 64, device=x.device)
    #
    #     x = self.conv1(x)
    #     x = self.conv2(x)
    #     x = self.conv3(x)
    #
    #     x_inp1 = self.inp1(x)
    #     x_inp2 = self.inp2(x)
    #     x_inp3 = self.inp3(x)
    #
    #     x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)
    #
    #     x = x.permute(0, 2, 1, 3)
    #     x = torch.reshape(x, (batch_size, x.shape[1], -1))
    #
    #     x, _ = self.lstm(x, (h0, c0))
    #     x = x[:, -1, :]
    #     x = self.fc1(x)
    #     forecast_y = torch.softmax(x, dim=1)
    #
    #     return forecast_y

    def forward(self, x):
        print(f"Input shape: {x.shape}")  # Initial input shape

        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, 64, device=x.device)
        c0 = torch.zeros(1, batch_size, 64, device=x.device)

        x = self.conv1(x)
        print(f"After conv1: {x.shape}")

        x = self.conv2(x)
        print(f"After conv2: {x.shape}")

        x = self.conv3(x)
        print(f"After conv3: {x.shape}")

        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)
        print(f"After inp1: {x_inp1.shape}, inp2: {x_inp2.shape}, inp3: {x_inp3.shape}")

        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)
        print(f"After concatenation: {x.shape}")

        x = x.permute(0, 2, 1, 3)  # (B, T, C, D)
        print(f"After permute: {x.shape}")

        x = x.mean(dim=-1, keepdim=False)

        x = torch.reshape(x, (batch_size, x.shape[1], -1))  # (B, T, C*D)
        print(f"After reshape: {x.shape}")

        if x.shape[-1] != 192:  # Debugging check
            raise ValueError(f"Expected input size 192, but got {x.shape[-1]}")

        x, _ = self.lstm(x, (h0, c0))
        print(f"After LSTM: {x.shape}")

        x = x[:, -1, :]
        x = self.fc1(x)
        forecast_y = torch.softmax(x, dim=1)

        return forecast_y