# %% CELL

import torch
import torch.nn as nn

from random import randrange
from typing import List, Union

class ProtoNet(nn.Module):
    def __init__(self, sidelen: int, out_features: int):
        super(ProtoNet, self).__init__()

        self.sidelen = sidelen
        self.out_features = out_features
        self.conv1 = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1)
        # 192 by 192 -> 194 by 194 WITH PADDING -> 192 by 192 WITH STRIDE 1
        self.linear = torch.nn.Linear(in_features=(sidelen*sidelen), out_features=out_features, bias=True)

        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.linear.weight)
        # nn.init.kaiming_uniform_(self.linear.bias)
    
    def forward(self, x):
        batch_dim = x.shape[0]
        # print('batch dim', batch_dim)

        x = self.conv1(x)
        x = self.linear(x.view(batch_dim, -1))

        return x

    def extra_repr(self):
        return 'sidelen {}, output, {}'.format(self.sidelen, self.out_features)

    # SAMPLE n WEIGHTS (WITH REPLACEMENT)
    def sample_weights(self, n: int) -> List:
        weights = self.linear.weight

        dims = weights.size()
        print('dims', dims)
        results = []
        for _ in range(n):
            index = []
            for d in dims:
                index.append(randrange(0, d))
            print(index)

            datum = weights[tuple(index)].item()
            results.append(datum)
            print(datum)

        return results
