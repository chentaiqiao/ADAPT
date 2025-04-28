import torch
from torch.distributions import Categorical, Normal
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm
from ADAPT.algorithms.mat.algorithm.ScoringNetwork import ScoringNetwork
from collections import deque

# Initialize lists with fixed capacity
dependency_vectors_list = deque(maxlen=100)  # Store up to 100 dependency vectors
final_P_list = deque(maxlen=100)  # Store up to 100 priority assignments
dependency_history = []
last_W = None
last_P = None
precomputed_priorities = []  # Cache for (dep_vectors, P*_t) pairs

def priority_scoring_network(dependency_vector, scoring_network, num_agents):

    scoring_network.eval()
    with torch.no_grad():
        # Ensure dependency vector is not empty
        if len(dependency_vector) == 0 or np.all(dependency_vector == 0):
            # Return random priority as fallback
            return np.random.randint(0, num_agents)
        
        # Keep non-self dependencies (exclude index where agent_i == agent_j)
        dependency_tensor = torch.from_numpy(dependency_vector).float()
        # Ensure correct shape: (num_agents - 1,)
        if dependency_tensor.shape[0] != num_agents - 1:
            # Pad or trim to match expected input
            dependency_tensor = dependency_tensor[:num_agents - 1]
            if dependency_tensor.shape[0] < num_agents - 1:
                dependency_tensor = F.pad(dependency_tensor, (0, num_agents - 1 - dependency_tensor.shape[0]), value=1e-6)
        
        dependency_tensor = dependency_tensor.to(next(scoring_network.parameters()).device)
        logits = scoring_network(dependency_tensor.unsqueeze(0))
        probs = torch.softmax(logits, dim=-1)
        predicted_priority = torch.argmax(probs, dim=-1).item()
    return predicted_priority

def discrete_autoregreesive_act(args, decoder, obs_rep, obs, relation_embed, relations, batch_size, n_agent, action_dim, tpdv,
                               available_actions=None, deterministic=False, dec_agent=False, time_step=0, scoring_network=None):
    global dependency_vectors_list, final_P_list, dependency_history, last_W, last_P, precomputed_priorities
    
    # Initialize difference_history with fixed length
    if not hasattr(args, 'max_history_length'):
        args.max_history_length = 1000
    global difference_history
    difference_history = deque(maxlen=args.max_history_length)
    
    def check_dependency(decoder, obs_rep, obs, available_actions, agent_i, agent_j, action_dim, tpdv, shifted_action, relation_embed, relations, dec_agent=False):
        original_logits = decoder(shifted_action, obs_rep, obs, relation_embed, attn_mask=relations, dec_agent=dec_agent)[:, agent_i, :]
        original_distri = Categorical(logits=original_logits)
        original_prob = original_distri.probs + 1e-10  # Add epsilon for numerical stability
    
        shifted_action_temp = shifted_action.clone()
        shifted_action_temp[:, agent_j, 1:] = F.one_hot(shifted_action[:, agent_j, 0].long(), num_classes=action_dim)
        logits_with_j = decoder(shifted_action_temp, obs_rep, obs, relation_embed, attn_mask=relations, dec_agent=dec_agent)[:, agent_i, :]
        with_j_distri = Categorical(logits=logits_with_j)
        with_j_prob = with_j_distri.probs + 1e-10  # Add epsilon for numerical stability
    
        kl_divergence_i_given_j = torch.sum(original_prob * torch.log(original_prob / with_j_prob))
        return kl_divergence_i_given_j.clamp(min=0)  # Ensure non-negative KL divergence

    def calculate_kl_divergence(decoder, obs_rep, obs, available_actions, agent_i, action_dim, tpdv, shifted_action, relation_embed, relations, n_agent):
        dependency_vector = np.zeros(n_agent)
        for agent_j in range(n_agent):
            if agent_i != agent_j:
                val_dependent = check_dependency(decoder, obs_rep, obs, available_actions, agent_i, agent_j, action_dim, tpdv, shifted_action, relation_embed, relations)
                dependency_vector[agent_j] = float(val_dependent)
        return dependency_vector

    def distributed_auction(dependency_vectors, scoring_network):
        n_agent = len(dependency_vectors)
        scores = []
        for i, w_i in enumerate(dependency_vectors):
            # Exclude self-dependency (i-th position)
            w_i_adjusted = np.delete(w_i, i)
            score = priority_scoring_network(w_i_adjusted, scoring_network, n_agent)
            scores.append([score] * n_agent)  # Duplicate score for each priority slot
        P = list(range(n_agent))
        prices = np.zeros(n_agent)
        unassigned = list(range(n_agent))
    
        while unassigned:
            bids = []
            for i in unassigned:
                utilities = np.array(scores[i]) - prices
                best_utility = np.max(utilities)
                best_priority = np.argmax(utilities)
                temp_utilities = np.copy(utilities)
                temp_utilities[best_priority] = -np.inf
                second_best_utility = np.max(temp_utilities)
                bid_price = best_utility - second_best_utility + args.epsilon
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
    
    epsilon = args.epsilon
    Threshold_Real_radio = args.Threshold_Real_radio 

    W = []
    shifted_action = torch.zeros((obs_rep.size(0), n_agent, action_dim + 1)).to(**tpdv)
    shifted_action[:, 0, 0] = 1
    output_action = torch.zeros((obs_rep.size(0), n_agent, 1), dtype=torch.long).to(**tpdv)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32).to(**tpdv)

    perm = torch.arange(n_agent).long()


    if len(dependency_vectors_list) < 100:
        if dependency_history:
            mu, std = norm.fit(dependency_history)
            Threshold_Real = mu + Threshold_Real_radio * std
        else:
            Threshold_Real = float('inf')
    
        for i in range(n_agent):
            w_i = calculate_kl_divergence(decoder, obs_rep, obs, available_actions, i, action_dim, tpdv, shifted_action, relation_embed, relations, n_agent)
            W.append(w_i)
        W = np.array(W)
        dependency_history.extend(W.flatten().tolist())
    
        if last_W is None:
            final_P = distributed_auction(W, scoring_network)
            dependency_vectors_list.append(W)
            final_P_list.append(final_P)
        else:
            wi_diff_norms = np.linalg.norm(last_W - W, axis=1)
            difference_history.extend(wi_diff_norms.tolist())
            last_W = W
            if (wi_diff_norms > Threshold_Real).any():
                final_P = distributed_auction(W, scoring_network)
                dependency_vectors_list.append(W)
                final_P_list.append(final_P)
            else:
                final_P = last_P
        last_P = final_P

        perm = torch.tensor(final_P, dtype=torch.long)

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
    """
    Return the list of (dependency_vectors, final_P) pairs.
    """
    global dependency_vectors_list, final_P_list
    return list(zip(dependency_vectors_list, final_P_list))

def discrete_parallel_act(decoder, obs_rep, obs, action, relation_embed, relations, batch_size, n_agent, action_dim, tpdv,
                          available_actions=None, dec_agent=False):
    one_hot_action = F.one_hot(action.squeeze(-1), num_classes=action_dim)
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

        distri = Normal(act_mean, action_std)
        action = act_mean if deterministic else distri.sample()
        action_log = distri.log_prob(action)

        output_action[:, i, :] = action
        output_action_log[:, i, :] = action_log
        if i + 1 < n_agent:
            shifted_action[:, i + 1, :] = action

    return output_action, output_action_log

def continuous_parallel_act(decoder, obs_rep, obs, action, relation_embed, relations, batch_size, n_agent, action_dim, tpdv, dec_agent=False):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim)).to(**tpdv)
    shifted_action[:, 1:, :] = action[:, :-1, :]

    act_mean = decoder(shifted_action, obs_rep, obs, relation_embed, attn_mask=relations, dec_agent=dec_agent)
    action_std = torch.sigmoid(decoder.log_std) * 0.5
    distri = Normal(act_mean, action_std)

    action_log = distri.log_prob(action)
    entropy = distri.entropy()
    return action_log, entropy