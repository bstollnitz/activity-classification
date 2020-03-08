import torch.nn as nn

class FeedForward(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        return nn.functional.log_softmax(x, dim=1)
