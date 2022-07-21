from typing import Any, Dict

import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(
        self,
        cfg: Dict[str, Any],
    ):
        super().__init__()
        self.cfg = cfg

        if self.cfg.activation == "relu":
            self.act_func = F.relu
        elif self.cfg.activation == "tanh":
            self.act_func = F.tanh
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")

        if self.cfg.batch_norm:
            self.fc1 = nn.Linear(self.cfg.in_features, 500)
            self.bn1 = nn.BatchNorm1d(500)
            self.fc2 = nn.Linear(500, 200)
            self.bn2 = nn.BatchNorm1d(200)
            self.fc3 = nn.Linear(200, 50)
            self.bn3 = nn.BatchNorm1d(50)
            self.fc4 = nn.Linear(50, self.cfg.classes)
        else:
            self.fc1 = nn.Linear(self.cfg.in_features, 500)
            self.fc2 = nn.Linear(500, 200)
            self.fc3 = nn.Linear(200, 50)
            self.fc4 = nn.Linear(50, self.cfg.classes)

    def forward(self, x):

        x = x.view(x.size(0), -1)

        if self.cfg.batch_norm:
            x = self.act_func(self.bn1(self.fc1(x)))
            x = self.act_func(self.bn2(self.fc2(x)))
            x = self.act_func(self.bn3(self.fc3(x)))
        else:
            x = self.act_func(self.fc1(x))
            x = self.act_func(self.fc2(x))
            x = self.act_func(self.fc3(x))

        return self.fc4(x)
