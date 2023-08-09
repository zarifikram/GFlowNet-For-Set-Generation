import torch
from torch import nn
from torch.nn.functional import relu, leaky_relu, one_hot
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
        self.size = state_dim # no one-hot encoding, so size and state_dim are the same
    
    
    def __call__(self, s, curr):

        left_edge = curr == 0
        right_edge = curr == self.size - 1
        
        # find the index of the chosen number
        chosen_num = s[torch.arange(len(s)), curr.long()]
        has_chosen = chosen_num != 0
     
        probs = 1 * torch.ones(len(s), self.num_actions)
        probs[has_chosen, : self.num_actions - 3] = one_hot(chosen_num[has_chosen].long() - 1, self.num_actions - 3).float()
        probs[~has_chosen, : self.num_actions - 3] = 0
        probs[left_edge, self.num_actions - 2] = 0 # if leftmost, surely didn't come from left
        probs[right_edge, self.num_actions - 3] = 0 # if rightmost, surely didn't come from right
        probs[:, -1] = 0 # disregard termination
        
        # # now normalize
        probs = probs / probs.sum(dim=1).view(-1, 1)
        return probs

