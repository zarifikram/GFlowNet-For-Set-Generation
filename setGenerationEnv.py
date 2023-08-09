import torch
from torch.nn.functional import one_hot
from gflownet.env import Env

class SetGenerationEnv(Env):
    # takes s (state) [bcz_size x size]
    # takes curr (current position) [bcz_size]
    # 0 means no number is assigned to that position. So initial state is torch.zeros(bcz_size, size)
    def __init__(self, size, number=4):
        self.size = size
        self.state_dim = size # no one-hot encoding, so size and state_dim are the same
        self.num_actions = number + 3 # actions, down, right, terminate
        
    def update(self, s, curr, actions):
        # actions is a one-hot vector
        # left is num_actions-3, right is num_actions-2, terminate is num_actions - 1
   
        left, right, chosen_num = actions == self.num_actions - 3, actions == self.num_actions - 2, actions < self.num_actions - 3

        curr[left] = curr[left] - 1
        curr[right] = curr[right] + 1
        
        s[chosen_num, curr[chosen_num].long()] = actions[chosen_num].float() + 1

        return s.float(), curr
    
    def mask(self, s, curr):
        mask = torch.ones(len(s), self.num_actions)

        chosen = s[torch.arange(len(s)), curr.long()] != 0
        # right_edge = curr == self.size - 1
        # left_edge = curr == 0
        # aka don't go left as all the slots to the left are filled
        # we want values to the left of each curr. So curr = [3, 2] and s = [[1, 2, 3, 4], [1, 2, 3, 4]]
        # then we want [[1, 2, 3], [1, 2]]

        
        # result = [(row[:c.item()] != 0).all()   for row, c in zip(s, curr.long())]
        # no_val_to_the_left = torch.stack(result)

        # result = [(row[c.item() + 1 :] != 0).all()   for row, c in zip(s, curr.long())]
        # no_val_to_the_right = torch.stack(result)

        left_edge = (curr == 0)
        right_edge = (curr == self.size - 1)
        

        # if each s[:, :] is filled, then terminate
        done = (s != 0).all(dim=1)
        notdone = ~done
       
        # mask[notdone, self.num_actions - 1] = 0
        mask[chosen, :self.num_actions - 3] = 0
        mask[left_edge, self.num_actions - 3] = 0
        mask[right_edge, self.num_actions - 2] = 0
        # mask[no_val_to_the_left, self.num_actions - 3] = 0
        # mask[no_val_to_the_right, self.num_actions - 2] = 0

        # only if terminated, do not hang around
        mask[done, self.num_actions - 3] = 0
        mask[done, self.num_actions - 2] = 0
        mask[done, :self.num_actions - 3] = 0
        
        return mask
        
    def reward(self, s):
        s = self.getStateForm(s)
       
        R0 = (s[:, 0] - s[:, 1]) ** 2 + (s[:, 2] - s[:, 3]) ** 2 + 1e-2
        # if any has 0 in it, R0 for that sample will be 1e-2
        R0[(s == 0).any(dim = 1)] = 1e-2
        # R0 = 5**(R0 / 13**2)
        # R0 = R0/10
        # R0[R0 < 1500] = 1e-8
        return R0.float() 
    
    def getStateForm(self, s):
        return s