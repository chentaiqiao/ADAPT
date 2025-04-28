import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ADAPT.utils.util import get_gard_norm, huber_loss, mse_loss, get_shape_from_obs_space, get_shape_from_act_space
from ADAPT.utils.valuenorm import ValueNorm
from ADAPT.algorithms.utils.util import check
from ADAPT.algorithms.utils.transformer_act import get_list, precomputed_priorities
from ADAPT.algorithms.mat.algorithm.MessageGenerationNetwork import MessageGenerationNetwork
from ADAPT.algorithms.mat.algorithm.ObservationReconstructionNetwork import ObservationReconstructionNetwork
from ADAPT.algorithms.mat.algorithm.ScoringNetwork import ScoringNetwork
from torch.distributions import Categorical, Normal
import torch.optim as optim
import logging
import json
from datetime import datetime



class MATTrainer:
    def __init__(self,
                 args,
                 policy,
                 num_agents,
                 device=torch.device("cpu"),
                 envs_info=None):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.num_agents = num_agents


        obs_space = envs_info.observation_space[0]
        self.obs_dim = get_shape_from_obs_space(obs_space)[0]
        act_space = envs_info.action_space[0]
        self.action_dim = act_space.n
        self.num_actions = self.action_dim
        # print(envs_info.observation_space[0],envs_info.action_space[0])

        self.latent_dim = args.latent_dim if hasattr(args, 'latent_dim') else 32
        self.message_dim = args.message_dim if hasattr(args, 'message_dim') else 24
        self.message_gen_network = MessageGenerationNetwork(
            input_dim=self.obs_dim + self.action_dim,
            latent_dim=self.latent_dim,
            message_dim=self.message_dim
        ).to(device)
        self.message_optimizer = optim.Adam(self.message_gen_network.parameters(), lr=args.lr)


        self.recon_network = ObservationReconstructionNetwork(
            message_dim=self.message_dim,
            obs_dim=self.obs_dim
        ).to(device)
        self.recon_optimizer = optim.Adam(self.recon_network.parameters(), lr=args.lr)


        self.variational_decoder = nn.Sequential(
            nn.Linear(self.message_dim + self.obs_dim + self.action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim)  
        ).to(device)
        self.decoder_optimizer = optim.Adam(self.variational_decoder.parameters(), lr=args.lr)

        # Initialize scoring network globally
        self.scoring_network = ScoringNetwork(input_dim=num_agents-1, output_dim=num_agents).to(device)
        self.scoring_optimizer = optim.Adam(self.scoring_network.parameters(), lr=args.edge_lr)

        self.update_step = 0
        self.precomputed_done = False
        
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        self.dec_actor = args.dec_actor
        self._use_bilevel = args.use_bilevel
        self._post_stable = args.post_stable
        self._use_post = args.post_stable
        self._post_ratio = args.post_ratio
        self.edge_lr = args.edge_lr
        
        if self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def compute_message_loss(self, encoded_obs, prev_actions, next_actions):
        # encoded_obs: [batch_size, obs_dim]
        # prev_actions, next_actions: [batch_size, num_actions] (one-hot)
        message, dist = self.message_gen_network(encoded_obs, prev_actions)  # [batch_size, message_dim]

        # τ_t^i = (o_t^i, a_{t-1}^i)
        trajectory = torch.cat([encoded_obs, prev_actions], dim=-1)  # [batch_size, obs_dim + num_actions]

        # q(a_t^i | c_t^i, τ_t^i)
        decoder_input = torch.cat([message, trajectory], dim=-1)  # [batch_size, message_dim + obs_dim + num_actions]
        action_logits = self.variational_decoder(decoder_input)  # [batch_size, num_actions]

        # D_KL(p(a_t^i | τ_t^i) || q(a_t^i | c_t^i, τ_t^i))
        log_q = F.log_softmax(action_logits, dim=-1)
        p_dist = Categorical(probs=next_actions + 1e-10)
        log_p = p_dist.log_prob(torch.argmax(next_actions, dim=-1))
        kl_div = F.kl_div(log_q, next_actions, reduction='batchmean')

        return -kl_div  

    def compute_reconstruction_loss(self, messages, encoded_obs):
        recon_obs = self.recon_network(messages)  # [batch_size, obs_dim]
        recon_loss = F.mse_loss(recon_obs, encoded_obs, reduction='mean')
        return recon_loss
        
    def train(self, buffer, step, total_step):
        advantages_copy = buffer.advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (buffer.advantages - mean_advantages) / (std_advantages + 1e-5)
        
        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info['message_loss'] = 0
        train_info['recon_loss'] = 0

        for i in range(self.ppo_epoch):
            data_generator = buffer.feed_forward_generator_transformer(advantages, self.num_mini_batch)
            
            for sample in data_generator:
                share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                adv_targ, available_actions_batch = sample

                encoded_obs = check(obs_batch).to(**self.tpdv)  # [batch_size, obs_dim]
                actions_batch_tensor = check(actions_batch).to(**self.tpdv)  # [batch_size, 1]
                actions_one_hot = F.one_hot(actions_batch_tensor.squeeze(-1).long(), num_classes=self.num_actions).float()  # [batch_size, num_actions]
                prev_actions = torch.zeros_like(actions_one_hot).to(**self.tpdv)  # [batch_size, num_actions]
                next_actions = actions_one_hot

                self.message_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()
                message_loss = self.compute_message_loss(encoded_obs, prev_actions, next_actions)
                message_loss.backward()
                self.message_optimizer.step()
                self.decoder_optimizer.step()

                messages, _ = self.message_gen_network(encoded_obs, prev_actions)  # [batch_size, message_dim]
                self.recon_optimizer.zero_grad()
                recon_loss = self.compute_reconstruction_loss(messages, encoded_obs)
                recon_loss.backward()
                self.recon_optimizer.step()

                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample, step, i, total_step=total_step)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()
                train_info['message_loss'] += message_loss.item()
                train_info['recon_loss'] += recon_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info


    def compute_optimal_priority(self, dependency_vectors):
        """
        Compute approximate optimal priority P*_t using weighted sorting.
        dependency_vectors: (N, N) array, w_t^i[j] for all i, j (including i=j).
        Returns: P, list of length N, approximate P*_t.
        """
        N = len(dependency_vectors)
        dep_vectors = torch.tensor(dependency_vectors, dtype=torch.float32)
        
        # Initialize priority list and remaining agents
        P = []
        remaining = list(range(N))
        
        # Precompute initial dependency sums: sum_j w_t^i[j]
        dep_sums = torch.zeros(N)
        for i in range(N):
            dep_sums[i] = dep_vectors[i].sum()
        
        # Greedy selection based on dependency sums
        while remaining:
            min_sum, best_agent = float('inf'), None
            for i in remaining:
                if dep_sums[i] < min_sum:
                    min_sum = dep_sums[i]
                    best_agent = i
            
            P.append(best_agent)
            remaining.remove(best_agent)
            
            # Update dependency sums for remaining agents
            for i in remaining:
                dep_sums[i] -= dep_vectors[i][best_agent]
        
        return P

    def precompute_priorities(self):
        """
        Compute P*_t for all dependency vectors in get_list() when length reaches 100.
        Store results in precomputed_priorities.
        """
        global precomputed_priorities
        samples = get_list()
        if len(samples) >= 100 and not self.precomputed_done:
            precomputed_priorities = []
            for dep_vectors, _ in samples:
                P_star = self.compute_optimal_priority(dep_vectors)
                precomputed_priorities.append((dep_vectors, P_star))
            self.precomputed_done = True
            print(f"Precomputed priorities for {len(precomputed_priorities)} dependency vectors")
            logging.info(f"Precomputed priorities for {len(precomputed_priorities)} dependency vectors")

    def train_priority_scoring_network(self, dependency_vectors, final_P):
        """
        Train scoring network using cached (dep_vectors, P*_t) pairs if available.
        Otherwise, compute P*_t on-the-fly.
        Loss: L_S(theta_S) = -1/N sum_i log(softmax(v_t^i)[P*_t[i]])
        """
        self.scoring_network.train()
        self.scoring_optimizer.zero_grad()
    
        # Use cached priorities if available
        global precomputed_priorities
        if precomputed_priorities:
            # Randomly select a cached (dep_vectors, P*_t) pair
            idx = np.random.randint(len(precomputed_priorities))
            dep_vectors, P_star = precomputed_priorities[idx]
        else:
            # Compute P*_t on-the-fly
            dep_vectors = np.array(dependency_vectors)
            if dep_vectors.shape != (self.num_agents, self.num_agents):
                raise ValueError(f"Expected dependency_vectors shape ({self.num_agents}, {self.num_agents}), got {dep_vectors.shape}")
            P_star = self.compute_optimal_priority(dep_vectors)
        
        # Prepare dependency vectors (exclude self-dependency)
        dep_vectors = np.array([dep_vectors[i, [j for j in range(self.num_agents) if j != i]] 
                              for i in range(self.num_agents)])
        dep_tensor = torch.tensor(dep_vectors, dtype=torch.float32).to(self.device)
        P_star_tensor = torch.tensor(P_star, dtype=torch.long).to(self.device)
    
        # Compute logits and softmax probabilities
        logits = self.scoring_network(dep_tensor)
        probs = torch.softmax(logits, dim=-1)
        
        # Compute loss: -log(softmax(v_t^i)[P*_t[i]])
        log_probs = torch.log(probs + 1e-10)
        loss = -torch.mean(log_probs[torch.arange(self.num_agents), P_star_tensor])
        

        
        loss.backward()
        self.scoring_optimizer.step()


    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                    self.clip_param)

        if self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    
    def ppo_update(self, sample, steps, index, total_step=0):
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                            obs_batch, 
                            rnn_states_batch, 
                            rnn_states_critic_batch, 
                            actions_batch, 
                            masks_batch, 
                            available_actions_batch,
                            active_masks_batch,
                            steps,
                            total_step,)
        # Actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_loss = (-torch.sum(torch.min(surr1, surr2),
                                      dim=-1,
                                      keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        # Critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        loss = policy_loss - dist_entropy * self.entropy_coef + value_loss * self.value_loss_coef

        if self._use_bilevel:
            if (index+1) % 5 == 0 and ((self._use_post and steps <= int(self._post_ratio * total_step)) or not self._use_post):
                self.policy.edge_optimizer.zero_grad()
            else:
                self.policy.optimizer.zero_grad()
        else:
            self.policy.optimizer.zero_grad()
            if (index+1) % 5 == 0:
                self.policy.edge_optimizer.zero_grad()
                
        loss.backward()

        if self._use_max_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(self.policy.transformer.model_parameters(), self.max_grad_norm)
        else:
            grad_norm = get_gard_norm(self.policy.transformer.model_parameters())
        
        if self._use_bilevel:
            if (index+1) % 5 == 0 and ((self._use_post and steps <= int(self._post_ratio * total_step)) or not self._use_post):
                self.policy.edge_optimizer.step()
            else:
                self.policy.optimizer.step()
        else:
            self.policy.optimizer.step()
        
        # Check if precomputation is needed
        samples = get_list()
        if len(samples) >= 100:
            self.precompute_priorities()
        
            # Train scoring network
            if samples:
                dep_vectors, final_P = samples[np.random.randint(len(samples))]
                self.train_priority_scoring_network(dep_vectors, final_P)
    

            self.update_step += 1
        
        return value_loss, grad_norm, policy_loss, dist_entropy, grad_norm, imp_weights   


    def prep_training(self):
        self.policy.train()
    
    def prep_rollout(self):
        self.policy.eval()