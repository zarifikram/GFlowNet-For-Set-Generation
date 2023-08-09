import torch


def trajectory_balance_loss(total_flow, rewards, fwd_probs, back_probs):
    """
    Computes the mean trajectory balance loss for a collection of samples. For
    more information, see Bengio et al. (2022): https://arxiv.org/abs/2201.13259
    
    Args:
        total_flow: The estimated total flow used by the GFlowNet when drawing
        the collection of samples for which the loss should be computed
        
        rewards: The rewards associated with the final state of each of the
        samples
        
        fwd_probs: The forward probabilities associated with the trajectory of
        each sample (i.e. the probabilities of the actions actually taken in
        each trajectory)
        
        back_probs: The backward probabilities associated with each trajectory
    """
   
    # we use the idea of log sum exp to avoid overflow
    lhs = torch.log(fwd_probs).sum(dim=1) + total_flow # we are actually getting logflow
    rhs = torch.log(back_probs).sum(dim=1) + torch.log(rewards)
    
    loss = (lhs - rhs)**2
    # loss[loss > 10000] = 1000
    # if (loss > 1000).any():
    #     print(f'WARNING: trajectory balance loss is large: {loss}. ')
    #     print(f"log total flow: {torch.log(total_flow)}")
    #     print(f"log rewards: {torch.log(rewards)}")
    #     print(f"log fwd probs: {torch.log(fwd_probs)}")
    #     print(f"log back probs: {torch.log(back_probs)}")
        
        
    return loss.mean()
