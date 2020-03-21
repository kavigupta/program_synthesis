import torch_geometric
import torch.nn as nn

class GCNConv(torch_geometric.nn.MessagePassing):
    """
    Copied (with minor changes) from the example
        in https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
    """
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization
        row, col = edge_index
        deg = torch_geometric.utils.degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-6: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x,
                              norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 6: Return new node embeddings.
        return aggr_out

class MultiGCNConv(nn.Module):
    def __init__(self, dim, layers):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(dim, dim) for _ in range(layers)])
        self.conv_activation = nn.Sigmoid()

    def forward(self, vertices, edges):
        for conv in self.convs[:-1]:
            vertices = conv(vertices, edges)
            vertices = self.conv_activation(vertices)
        if len(self.convs) > 0:
            vertices = self.convs[-1](vertices, edges)
        return vertices
