import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, in_features):
        super(BaseModel, self).__init__()

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
        )

        self.output_char_layers = [
            nn.Linear(in_features=512, out_features=26),
            nn.Linear(in_features=512, out_features=26),
            nn.Linear(in_features=512, out_features=26),
            nn.Linear(in_features=512, out_features=26),
            nn.Linear(in_features=512, out_features=26)
        ]

        self.flatten = nn.Flatten(start_dim=0)
        self.activation = nn.ReLU()

    def forward(self, x):
        output = self.flatten(x)
        output = self.linear_layers(output)

        outputs = torch.empty((5, 26))
        for i, layer in enumerate(self.output_char_layers):
            outputs[i] = layer(output)
        
        return outputs
    