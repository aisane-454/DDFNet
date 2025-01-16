import torch
from torch import nn

class BinaryFocalLoss(nn.Module):

    def __init__(self, alpha=0.6, gamma=2.0, epsilon=1.e-9, smoothing=0.2):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.smoothing = smoothing

    def forward(self, input, target):
        """
        Args:
            input: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        multi_hot_key = (1-self.smoothing) * target + self.smoothing / 2
        logits = input
        logits = torch.sigmoid(logits)
        zero_hot_key = 1 - multi_hot_key
        loss = -self.alpha * multi_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
        loss += -(1 - self.alpha) * zero_hot_key * torch.pow(logits, self.gamma) * (1 - logits + self.epsilon).log()
        return loss.mean()

if __name__ == '__main__':
    m = nn.Sigmoid()
    loss = BinaryFocalLoss()
    input = torch.randn(3, requires_grad=True)
    target = torch.empty(3).random_(2)
    output = loss(m(input), target)
    print("loss:", output)
    output.backward()
