import torch
from torch.distributions import Categorical, Normal
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm


dependency_vectors_list,final_P_list,dependency_history,difference_history,last_W,last_P=[],[],[],[],[],[]
class ScoringNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=None):
        super(ScoringNetwork, self).__init__()
        if output_dim is None:
            output_dim = input_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )
    
    def forward(self, x):
        logits = self.network(x)  # Use the network layers
        return logits



def priority_scoring_network(dependency_vector, scoring_network):

    scoring_network.eval()
    with torch.no_grad():
        dependency_tensor = torch.from_numpy(dependency_vector).float().to(next(scoring_network.parameters()).device)
        logits = scoring_network(dependency_tensor.unsqueeze(0)) 
        probs = torch.softmax(logits, dim=-1)
        predicted_priority = torch.argmax(probs, dim=-1).item()
    
    return predicted_priority


def discrete_autoregreesive_act(args, decoder, obs_rep, obs, relation_embed, relations, batch_size, n_agent, action_dim, tpdv,
                                available_actions=None, deterministic=False, dec_agent=False, time_step=0):
    global dependency_vectors_list,final_P_list,dependency_history,difference_history,last_W,last_P
    
    def check_dependency(decoder, obs_rep, obs, available_actions, agent_i, agent_j, action_dim, tpdv, shifted_action,relation_embed, relations,dec_agent=False):#calculate w_i[j]
        original_logits = decoder(shifted_action, obs_rep, obs, relation_embed, attn_mask=relations, dec_agent=dec_agent)[:, agent_i, :]
        original_distri = Categorical(logits=original_logits)
        original_prob = original_distri.probs
    
        shifted_action_temp = shifted_action.clone()
        shifted_action_temp[:, agent_i, 1:] = F.one_hot(shifted_action[:, agent_j, 0].long(), num_classes=action_dim)
        logits_with_j = decoder(shifted_action_temp, obs_rep, obs, relation_embed, attn_mask=relations, dec_agent=dec_agent)[:, agent_i, :]
        with_j_distri = Categorical(logits=logits_with_j)
        with_j_prob = with_j_distri.probs
    
        kl_divergence_i_given_j = torch.sum(torch.where(original_prob != 0, torch.log(original_prob / with_j_prob), 0))
    
        return kl_divergence_i_given_j 
    
    
    def calculate_kl_divergence(decoder, obs_rep, obs, available_actions, agent_i, action_dim, tpdv, shifted_action, relation_embed, relations, n_agent):

        dependency_vector = np.zeros(n_agent)
    
        for agent_j in range(n_agent):
            if agent_i != agent_j:
                val_dependent = check_dependency(decoder, obs_rep, obs, available_actions, agent_i, agent_j, action_dim, tpdv, shifted_action, relation_embed, relations)
                dependency_vector[agent_j] = float(val_dependent)
            else:
              dependency_vector[agent_j] = 0.0
    
        return dependency_vector

    
            
    def calculate_cost(P, W):

      cost = 0
      N = len(P)
      for i in range(N):
        for j in range(N):
          if P[i] < P[j]:
            cost += W[i][j]
      return cost
    
    
    def distributed_auction(dependency_vectors):
        n_agent = len(dependency_vectors)
        scoring_network = ScoringNetwork(input_dim=n_agent)
        scoring_network.eval()  
        scores = [priority_scoring_network(w_i, scoring_network) for w_i in dependency_vectors]
        P = list(range(n_agent))  
        prices = np.zeros(n_agent)
        unassigned = list(range(n_agent))
    
        while unassigned:
            bids = []
            for i in unassigned:
                utilities = scores[i] - prices
                best_utility = np.max(utilities)
                best_priority = np.argmax(utilities)
    
                temp_utilities = np.copy(utilities)
                temp_utilities[best_priority] = -np.inf
                second_best_utility = np.max(temp_utilities)
                bid_price = best_utility - second_best_utility + epsilon
                bids.append((i, best_priority, bid_price))
    
            winning_bids = {}
            for i, priority, bid_price in bids:
                if priority not in winning_bids or bid_price > winning_bids[priority][1]:
                    winning_bids[priority] = (i, bid_price)
    
            newly_assigned = []
            for priority, (winner, bid_price) in winning_bids.items():
                P[priority] = winner
                prices[priority] = bid_price
                if winner in unassigned:
                    unassigned.remove(winner)
                newly_assigned.append(winner)
    
        sorted_indices = sorted(range(len(P)), key=lambda k: P[k])
        P_mapped = [0] * n_agent
        for i, original_index in enumerate(sorted_indices):
            P_mapped[original_index] = i
    
        return P_mapped
    
    def local_optimization(P, dependency_vectors, Threshold_loc_radio):
        n_agent = len(P)
        P_optimized = P.copy()

        
    
        for i in range(n_agent):
            for j in range(n_agent):

                dependency_history.append(dependency_vectors[i][j])
                if len(dependency_history) > args.save_interval:
                    dependency_history.pop(0) 

                mu, std = norm.fit(dependency_history)
                threshold_loc = mu + Threshold_loc_radio * std
                if dependency_vectors[i][j] > threshold_loc and P_optimized.index(i) < P_optimized.index(j):
                    proposed_P = P_optimized.copy()
                    idx_i = proposed_P.index(i)
                    idx_j = proposed_P.index(j)
    
                    if idx_i < idx_j:
                        proposed_P.pop(idx_i)
                        proposed_P.insert(idx_j,i)
                    
                    # cost(p')<cost(p)?
                    current_cost = calculate_cost(P_optimized, dependency_vectors)
                    proposed_cost = calculate_cost(proposed_P, dependency_vectors)
    
                    if proposed_cost < current_cost:
                        P_optimized = proposed_P
    
        return P_optimized
    
    epsilon = args.epsilon  
    Threshold_loc_radio = args.Threshold_loc_radio 
    Threshold_Real_radio = args.Threshold_Real_radio 

    W = np.zeros((n_agent, n_agent))

    shifted_action = torch.zeros((obs_rep.size(0), n_agent, action_dim + 1)).to(**tpdv)
    shifted_action[:, 0, 0] = 1
    output_action = torch.zeros((obs_rep.size(0), n_agent, 1), dtype=torch.long).to(**tpdv)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32).to(**tpdv)

    perm = torch.arange(n_agent).long()



    mu, std = norm.fit(dependency_history)
    Threshold_Real = mu + Threshold_Real_radio * std

    dependency_vectors = []
    for i in range(n_agent):
        w_i = calculate_kl_divergence(decoder, obs_rep, obs, available_actions, i, action_dim, tpdv, shifted_action, relation_embed, relations, n_agent)
        dependency_vectors.append(w_i)
        W[i] = w_i

    if last_W==[]:
        initial_P = distributed_auction(dependency_vectors)
        final_P = local_optimization(initial_P, W, Threshold_loc_radio)
        dependency_vectors=np.stack(dependency_vectors, axis=0)
        if len(final_P_list) > args.ppo_epoch:
            dependency_vectors_list.pop(0)
            final_P_list.pop(0)
        dependency_vectors_list.append(dependency_vectors)
        final_P_list.append(final_P)
    else:
        wi_diff_norms = np.linalg.norm(last_W - W, axis=1)
        difference_history.extend(wi_diff_norms.tolist())
        if len(difference_history) > args.save_interval:
            difference_history.pop(0) 
        last_W=W  
        if (wi_diff_norms > Threshold_Real).any():#if any agent need adjust ：run distributed_auction+local_optimization
            initial_P = distributed_auction(dependency_vectors)
            final_P = local_optimization(initial_P, W, Threshold_loc_radio)
            # print("Initial Priority:", initial_P)
            # print("Final Priority:", final_P)
            # For scoring net training
            dependency_vectors=np.stack(dependency_vectors, axis=0)
            if len(final_P_list) > args.ppo_epoch:
                dependency_vectors_list.pop(0)
                final_P_list.pop(0)
            dependency_vectors_list.append(dependency_vectors)
            final_P_list.append(final_P)
        else:#run real-time ajust
            final_P = local_optimization(last_P, W, Threshold_loc_radio)
    last_P=final_P
    perm = final_P

    
    
    # action sequence 
    for i in range(n_agent):
        logit = decoder(shifted_action, obs_rep, obs, relation_embed, attn_mask=relations, dec_agent=dec_agent)[:, perm[i], :]
        if available_actions is not None:
            logit[available_actions[:, perm[i], :] == 0] = -1e10
        distri = Categorical(logits=logit)
        action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
        action_log = distri.log_prob(action)

        output_action[:, perm[i], :] = action.unsqueeze(-1)
        output_action_log[:, perm[i], :] = action_log.unsqueeze(-1)
        if i + 1 < n_agent:
            shifted_action[:, perm[i + 1], 1:] = F.one_hot(action, num_classes=action_dim)

    
    return output_action, output_action_log

def get_list():
    global dependency_vectors_list,final_P_list
    return dependency_vectors_list,final_P_list
   
    
def discrete_parallel_act(decoder, obs_rep, obs, action, relation_embed, relations, batch_size, n_agent, action_dim, tpdv,
                          available_actions=None, dec_agent=False):
    one_hot_action = F.one_hot(action.squeeze(-1), num_classes=action_dim)  # (batch, n_agent, action_dim)
    shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1)).to(**tpdv)
    shifted_action[:, 0, 0] = 1
    shifted_action[:, 1:, 1:] = one_hot_action[:, :-1, :]
    logit = decoder(shifted_action, obs_rep, obs, relation_embed, attn_mask=relations, dec_agent=dec_agent)
    if available_actions is not None:
        logit[available_actions == 0] = -1e10

    distri = Categorical(logits=logit)
    action_log = distri.log_prob(action.squeeze(-1)).unsqueeze(-1)
    entropy = distri.entropy().unsqueeze(-1)
    return action_log, entropy


def continuous_autoregreesive_act(decoder, obs_rep, obs, relation_embed, relations, batch_size, n_agent, action_dim, tpdv,
                                  deterministic=False, dec_agent=False):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim)).to(**tpdv)
    output_action = torch.zeros((batch_size, n_agent, action_dim), dtype=torch.float32)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)

    for i in range(n_agent):
        act_mean = decoder(shifted_action, obs_rep, obs, relation_embed, attn_mask=relations, dec_agent=dec_agent)[:, i, :]
        action_std = torch.sigmoid(decoder.log_std) * 0.5

        # log_std = torch.zeros_like(act_mean).to(**tpdv) + decoder.log_std
        # distri = Normal(act_mean, log_std.exp())
        distri = Normal(act_mean, action_std)
        action = act_mean if deterministic else distri.sample()
        action_log = distri.log_prob(action)

        output_action[:, i, :] = action
        output_action_log[:, i, :] = action_log
        if i + 1 < n_agent:
            shifted_action[:, i + 1, :] = action

        # print("act_mean: ", act_mean)
        # print("action: ", action)

    return output_action, output_action_log


def continuous_parallel_act(decoder, obs_rep, obs, action, relation_embed, relations, batch_size, n_agent, action_dim, tpdv, dec_agent=False):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim)).to(**tpdv)
    shifted_action[:, 1:, :] = action[:, :-1, :]

    act_mean = decoder(shifted_action, obs_rep, obs, relation_embed, attn_mask=relations, dec_agent=dec_agent)
    action_std = torch.sigmoid(decoder.log_std) * 0.5
    distri = Normal(act_mean, action_std)

    # log_std = torch.zeros_like(act_mean).to(**tpdv) + decoder.log_std
    # distri = Normal(act_mean, log_std.exp())

    action_log = distri.log_prob(action)
    entropy = distri.entropy()
    return action_log, entropy
