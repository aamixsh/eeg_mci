import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, num_channels=19, num_psd=26, out_dim=2, conf=[64, 32, 16]):
        super(MLP, self).__init__()
        
        num_in = num_channels * num_psd
        
        layers = []
        for layer_size in conf:
            layers.append(nn.Linear(num_in, layer_size))
            layers.append(nn.ReLU())
            num_in = layer_size
        layers.append(nn.Linear(num_in, out_dim))
        
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        out = x.view(x.shape[0], -1)
        out = self.fc_layers(out)
        return out

class CNN(nn.Module):
    def __init__(self, num_channels=19, num_psd=26, kernel=3, out_dim=2):
        super(CNN, self).__init__()
        if num_channels == 19:
            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 2, kernel),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(2, 4, kernel),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(4, 8, kernel),
                nn.ReLU(),
                nn.Conv2d(8, 8, 1),
                nn.ReLU()
            )
            self.fc_layers = nn.Sequential(
                nn.Linear(8 * 3, 8),
                nn.ReLU(),
                nn.Linear(8, out_dim)
            )
        else:
            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 2, kernel),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(2, 4, kernel),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(4, 8, kernel-1),
                nn.ReLU(),
                nn.Conv2d(8, 8, 1),
                nn.ReLU()
            )
            self.fc_layers = nn.Sequential(
                nn.Linear(8 * 4, 8),
                nn.ReLU(),
                nn.Linear(8, out_dim)
            )

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(x.shape[0], -1)
        out = self.fc_layers(out)
        return out
