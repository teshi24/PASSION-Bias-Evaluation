from torch import nn


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features."""

    def __init__(
        self,
        dim: int,
        num_labels: int = 1000,
        use_dropout_in_head: bool = False,
        dropout_rate: float = 0.3,
        large_head: bool = True,
        use_bn: bool = False,
        log_softmax: bool = False,
    ):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.large_head = large_head
        self.use_bn = use_bn
        self.log_softmax = log_softmax
        self.use_dropout_in_head = use_dropout_in_head

        if self.use_dropout_in_head:
            self.dropout = nn.Dropout(dropout_rate)
        if self.use_bn:
            self.bn = nn.BatchNorm1d(dim)

        if self.large_head:
            self.linear = nn.Linear(dim, 128)
            self.linear.weight.data.normal_(mean=0.0, std=0.01)
            self.linear.bias.data.zero_()
            self.relu = nn.ReLU()

            self.dropout2 = nn.Dropout(dropout_rate)
            if self.use_bn:
                self.bn2 = nn.BatchNorm1d(128)

            self.linear2 = nn.Linear(128, num_labels)
            self.linear2.weight.data.normal_(mean=0.0, std=0.01)
            self.linear2.bias.data.zero_()
        else:
            self.linear = nn.Linear(dim, num_labels)
            self.linear.weight.data.normal_(mean=0.0, std=0.01)
            self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)
        # dropout
        if self.use_dropout_in_head:
            x = self.dropout(x)
        if self.use_bn:
            x = self.bn(x)
        # 1. linear layer
        x = self.linear(x)
        # smaller version of head
        if self.large_head:
            x = self.relu(x)
            x = self.dropout2(x)
            if self.use_bn:
                x = self.bn2(x)
            # 2. linear layer
            x = self.linear2(x)
        # output
        if self.log_softmax:
            return nn.LogSoftmax(dim=1)(x)
        else:
            return x
