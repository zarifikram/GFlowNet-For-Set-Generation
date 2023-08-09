import torch
from torch.nn.functional import one_hot
from gflownet.env import Env

class SetGenerationEnv(Env):
    # takes s (state) [bcz_size x size x num_actions]
    # takes curr (current position) [bcz_size]
    def __init__(self, size, number=4):
        self.size = size
        self.state_dim = size*number # one-hot encoding
        self.num_actions = number + 3 # actions, down, right, terminate
        
    def update(self, s, curr, actions):
        # actions is a one-hot vector
        # left is num_actions-3, right is num_actions-2, terminate is num_actions - 1
   
        left, right, chosen_num = actions == self.num_actions - 3, actions == self.num_actions - 2, actions < self.num_actions - 3

        curr[left] = curr[left] - 1
        curr[right] = curr[right] + 1
        k = s.clone()
  
        k.view(len(s), self.size, self.num_actions - 3)[chosen_num, curr[chosen_num].long(), actions[chosen_num].long()] = 1

        return k.float(), curr
    
    def mask(self, s, curr):
        mask = torch.ones(len(s), self.num_actions)
        k = s.clone()
        k[k < 0.2] = 0
        try:
            # print("Curr", curr)
            # print(k)
            # print(k.view(len(k), self.size, self.num_actions - 3).sum(dim=-1))
            # print(k.view(len(k), self.size, self.num_actions - 3).sum(dim=-1)[torch.arange(len(k)), curr.long()])
            chosen = k.view(len(k), self.size, self.num_actions - 3).sum(dim=-1)[torch.arange(len(k)), curr.long()] >= 1
        except IndexError:
            print(k.view(len(k), self.size, self.num_actions - 3).sum(dim=-1))
            print("Curr", curr)
            print("S", k)
            raise
        right_edge = curr == self.size - 1
        left_edge = curr == 0

        # if each k[:, :] has a 1, then it should be terminated (so, no left/right)
        done = k.sum(-1) == self.size
        notdone = k.sum(-1) != self.size
       
        mask[notdone, self.num_actions - 1] = 0
        mask[chosen, :self.num_actions - 3] = 0
        mask[left_edge, self.num_actions - 3] = 0
        mask[right_edge, self.num_actions - 2] = 0

        # only if terminated, do not hang around
        mask[done, self.num_actions - 3] = 0
        mask[done, self.num_actions - 2] = 0
        return mask
        
    def reward(self, s):
        k = self.getStateForm(s)
        R0 =  (k[:, 0] - k[:, 1]) ** 2 + (k[:, 2] - k[:, 3]) ** 2 + 1e-2
        R0 = 5**(R0 / 13**2)
        # R0 = R0/10
        # R0[R0 < 1500] = 1e-8
        return R0.float() 
    
    def getStateForm(self, s):
        k = s.clone()
        k[k < 0.2] = 0
        return k.view(len(k), self.size, self.num_actions - 3).argmax(dim=2) + 1