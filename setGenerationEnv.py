import torch
from torch.nn.functional import one_hot
from gflownet.env import Env

class SetGenerationEnv(Env):
    # takes s (state) [bcz_size x size]
    # 0 means no number is assigned to that position. So initial state is torch.zeros(bcz_size, size)
    def __init__(self, size, number=4):
        self.size = size
        self.state_dim = size # no one-hot encoding, so size and state_dim are the same
        self.num_actions = 2*number + 1 # [k added to left, k added to right, terminate action] k being one of the numbers
        self.number = number
        
    def update(self, s, actions):
        # left is 0 : num_acttions - 1, right is num_actions : 2*num_actions - 1

        left, right = actions < self.number , (actions >= self.number) & (actions < 2*self.number)

        # left means we shift everything to right and add the number to the left at 0th position
        s[left, 1:] = s[left, :-1]
        s[left, 0] = (actions[left] % (self.number) + 1).float()

        # right means we add the number at the leftmost empty position
        # we find the leftmost empty position by finding the first 0 in each row
        # and then adding the number there
        s[right, (s[right] == 0).long().argmax(dim = 1)] = (actions[right] % (self.number) + 1).float()

        return s.float()
    
    def mask(self, s):
        mask = torch.ones(len(s), self.num_actions)

        has_terminated = (s != 0).all(dim = 1)
        # everything but last action is masked
        mask[has_terminated, :-1] = 0

        # else only thelast action is masked
        mask[~has_terminated, -1] = 0
        return mask
        
    def reward(self, s):
        s = self.getStateForm(s)
       
        R0 = (s[:, 0] - s[:, 1]) ** 2 + (s[:, 2] - s[:, 3]) ** 2 + 1e-2
        R0 = 5**(R0 / (self.number - 2)**2)
        # R0 = R0/10
        # R0[R0 < 1500] = 1e-8
        return R0.float() 
    
    def getStateForm(self, s):
        return s