import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import BatchNorm
from torch_geometric.nn import GatedGraphConv, global_add_pool


class GNN(torch.nn.Module):

    def __init__(self, num_features, num_actions):
        super(GNN, self).__init__()
        self.conv1 = GatedGraphConv(out_channels=num_features, num_layers=4, aggr='add')
        self.conv2 = GatedGraphConv(out_channels=64, num_layers=4, aggr='add')
        self.conv3 = GatedGraphConv(out_channels=128, num_layers=4, aggr='add')

        self.fc1 = Linear(in_features=128, out_features=256)
        self.fc2 = Linear(in_features=256, out_features=num_actions)

        self.bn1 = BatchNorm(num_features)
        self.bn2 = BatchNorm(64)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.elu(self.conv1(x, edge_index))
        x = self.bn1(x)

        x = F.elu(self.conv2(x, edge_index))
        x = self.bn2(x)

        x = F.elu(self.conv3(x, edge_index))

        x = global_add_pool(x, batch)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)

        return F.log_softmax(self.fc2(x), dim=-1)
