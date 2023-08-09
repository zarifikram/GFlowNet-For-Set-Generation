import torch
from torch import nn
from torch.nn.functional import relu, leaky_relu
from torch.nn.functional import softmax

class ForwardPolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_actions):
        super().__init__()
        self.dense1 = nn.Linear(state_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, num_actions)
    
    def forward(self, s):
        x = self.dense1(s)
        if torch.isnan(x).any() or (x.sum(1).unsqueeze(1) == 0).any():
            print(f"probs from policy: {x}")
            print(f"s: {s}")
        x = leaky_relu(x)
        x = self.dense2(x)
        return softmax(x, dim=1)
    
class BackwardPolicy:
    def __init__(self, state_dim, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.size = int(state_dim / (num_actions - 3))
    
    
    def __call__(self, s, curr):
        # if rightmost, surely didn't come from right
        # if leftmost, surely didn't come from left

        left_edge = curr == 0
        right_edge = curr == self.size - 1
        # received_choice = s.view(len(s), self.size, self.num_actions - 3).sum(dim=-1)[torch.arange(len(s)), curr.long()] == 1

        # find the index of the chosen number
        k = s.clone()
        k[k < 0.2] = 0
        chosen_num = k.view(len(k), self.size, self.num_actions - 3)[torch.arange(len(k)), curr.long()]

        probs = 1 * torch.ones(len(k), self.num_actions)
        probs[:, :self.num_actions - 3] = chosen_num
        probs[left_edge, self.num_actions - 2] = 0
        probs[right_edge, self.num_actions - 3] = 0
        probs[:, -1] = 0 # disregard termination
        
        # now normalize
        probs = probs / probs.sum(dim=1).view(-1, 1)
        return probs

