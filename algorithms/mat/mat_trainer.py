import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ADAPT.utils.util import get_gard_norm, huber_loss, mse_loss
from ADAPT.utils.valuenorm import ValueNorm
from ADAPT.algorithms.utils.util import check
from ADAPT.algorithms.utils.transformer_act import *
from torch.distributions import Categorical, Normal
import torch.optim as optim



class MATTrainer:
    """
    Trainer class for MAT to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 num_agents,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.num_agents = num_agents

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
        self._post_ratio = args.post_ratio
        self.edge_lr=args.edge_lr
        
        if self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """

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

        # if self._use_value_active_masks and not self.dec_actor:
        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def train_priority_scoring_network(self,dependency_vectors, final_P, n_agents , learning_rate):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        # print(f"Using device: {device}")
        torch.cuda.empty_cache()
    
        # Convert dependency_vectors to tensor and ensure requires_grad=True
        dependency_vectors = np.array(dependency_vectors)  # Convert list to NumPy array
        dependency_vectors = torch.tensor(dependency_vectors).float().to(device)
        dependency_vectors.requires_grad_(True)
        # print(f"dependency_vectors requires_grad: {dependency_vectors.requires_grad}")
        # print(f"dependency_vectors device: {dependency_vectors.device}")
        # print(f"dependency_vectors dtype: {dependency_vectors.dtype}")
    
        final_P_tensor = torch.tensor(final_P, dtype=torch.long).to(device)
        final_P_tensor.requires_grad_(False)
  
    
        scoring_network = ScoringNetwork(input_dim=n_agents, output_dim=n_agents).to(device)
        for param in scoring_network.parameters():
            param.requires_grad = True
    
        optimizer = optim.Adam(scoring_network.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
    
        scoring_network.train()
        optimizer.zero_grad()
        logits = scoring_network(dependency_vectors)

        loss = criterion(logits, final_P_tensor)
        
        with torch.autograd.detect_anomaly():
            loss.backward()
        optimizer.step()


    def ppo_update(self, sample, steps, index, total_step=0):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic update.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
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
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_loss = (-torch.sum(torch.min(surr1, surr2),
                                      dim=-1,
                                      keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        loss = policy_loss - dist_entropy * self.entropy_coef + value_loss * self.value_loss_coef

        # self.policy.optimizer.zero_grad()
        if self._use_bilevel:
            if (index+1) % 5 == 0 and ((self._post_stable and steps <= int(self._post_ratio * total_step)) or not self._post_stable):
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
            if (index+1) % 5 == 0 and ((self._post_stable and steps <= int(self._post_ratio * total_step)) or not self._post_stable):
                self.policy.edge_optimizer.step()
            else:
                self.policy.optimizer.step()
        else:
            self.policy.optimizer.step()
        if (index+1) % 5 == 0:
            #---- Train scoring network ----
            dependency_vectors_list,final_P_list=get_list()
            for item in range(len(dependency_vectors_list)):
                self.train_priority_scoring_network(
                    dependency_vectors_list[item],
                    final_P_list[item],
                    self.num_agents, 
                    learning_rate=self.edge_lr
                )

            
        

        return value_loss, grad_norm, policy_loss, dist_entropy, grad_norm, imp_weights

    def train(self, buffer, step, total_step):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
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

        for i in range(self.ppo_epoch):
            data_generator = buffer.feed_forward_generator_transformer(advantages, self.num_mini_batch)
            
            for sample in data_generator:

                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample, step, i, total_step=total_step)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    def prep_training(self):
        self.policy.train()

    def prep_rollout(self):
        self.policy.eval()
