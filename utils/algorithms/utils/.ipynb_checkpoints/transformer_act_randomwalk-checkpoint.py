import torch
from torch.distributions import Categorical, Normal
from torch.nn import functional as F
import numpy as np

def check_dependency(decoder, obs_rep, obs, available_actions, agent_i, agent_j, action_dim, tpdv, shifted_action,relation_embed, relations,dec_agent=False):#检查i对j是否有依赖，有返回true
    # 保存原始动作分布
    original_logits = decoder(shifted_action, obs_rep, obs, relation_embed, attn_mask=relations, dec_agent=dec_agent)[:, agent_i, :]
    original_distri = Categorical(logits=original_logits)
    original_prob = original_distri.probs

    # 模拟智能体j对智能体i的影响
    shifted_action_temp = shifted_action.clone()
    shifted_action_temp[:, agent_i, 1:] = F.one_hot(shifted_action[:, agent_j, 0].long(), num_classes=action_dim)
    logits_with_j = decoder(shifted_action_temp, obs_rep, obs, relation_embed, attn_mask=relations, dec_agent=dec_agent)[:, agent_i, :]
    with_j_distri = Categorical(logits=logits_with_j)
    with_j_prob = with_j_distri.probs

    # 模拟智能体i对智能体j的影响
    shifted_action_temp = shifted_action.clone()
    shifted_action_temp[:, agent_j, 1:] = F.one_hot(shifted_action[:, agent_i, 0].long(), num_classes=action_dim)
    logits_with_i = decoder(shifted_action_temp, obs_rep, obs, relation_embed, attn_mask=relations, dec_agent=dec_agent)[:, agent_j, :]
    with_i_distri = Categorical(logits=logits_with_i)
    with_i_prob = with_i_distri.probs

    # 计算KL散度来评估依赖性
    kl_divergence_i_given_j = torch.sum(torch.where(original_prob != 0, torch.log(original_prob / with_j_prob), 0))
    kl_divergence_j_given_i = torch.sum(torch.where(with_i_prob != 0, torch.log(with_i_prob / original_prob), 0))

    # 如果j对i的影响大于i对j的影响，则i依赖于j
    return kl_divergence_i_given_j > kl_divergence_j_given_i

def discrete_autoregreesive_act(args, decoder, obs_rep, obs, relation_embed, relations, batch_size, n_agent, action_dim, tpdv,
                                available_actions=None, deterministic=False, dec_agent=False,time_step=0):#学习最优顺序，顺序更行
    

    
    shifted_action = torch.zeros((obs_rep.size(0), n_agent, action_dim + 1)).to(**tpdv)
    shifted_action[:, 0, 0] = 1
    output_action = torch.zeros((obs_rep.size(0), n_agent, 1), dtype=torch.long).to(**tpdv)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32).to(**tpdv)
    
    perm = torch.arange(n_agent).long()
    
    
    
    # 使用改正的新的顺序进行决策
    for i in range(n_agent):
        logit = decoder(shifted_action, obs_rep, obs, relation_embed, attn_mask=relations, dec_agent=dec_agent)[:, perm[i], :]
        if available_actions is not None:
            logit[available_actions[:, perm[i], :] == 0] = -1e10
        distri = Categorical(logits=logit)
        action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
        action_log = distri.log_prob(action)

        output_action[:, perm[i], :] = action.unsqueeze(-1)
        output_action_log[:, perm[i], :] = action_log.unsqueeze(-1)
        # if i + 1 < n_agent:
            # shifted_action[:, perm[i + 1], 1:] = F.one_hot(action, num_classes=action_dim)

    return output_action, output_action_log
    

# def check_dependency_batch(decoder, obs_rep, obs, available_actions, agent_pairs, action_dim, tpdv, shifted_action):
    # # 获取所有i和j的索引
    # i_indices = agent_pairs[:, 0]
    # j_indices = agent_pairs[:, 1]

    # # 保存原始动作分布
    # original_logits = decoder(shifted_action, obs_rep, obs)[:, i_indices, :]
    # original_distri = Categorical(logits=original_logits)
    # original_prob = original_distri.probs

    # # 模拟智能体j对智能体i的影响
    # shifted_action_temp = shifted_action.clone()
    # shifted_action_temp[:, i_indices, 1:] = F.one_hot(shifted_action[:, j_indices, 0].long(), num_classes=action_dim).float()
    # logits_with_j = decoder(shifted_action_temp, obs_rep, obs)[:, i_indices, :]
    # with_j_distri = Categorical(logits=logits_with_j)
    # with_j_prob = with_j_distri.probs

    # # 模拟智能体i对智能体j的影响
    # shifted_action_temp = shifted_action.clone()
    # shifted_action_temp[:, j_indices, 1:] = F.one_hot(shifted_action[:, i_indices, 0].long(), num_classes=action_dim).float()
    # logits_with_i = decoder(shifted_action_temp, obs_rep, obs)[:, j_indices, :]
    # with_i_distri = Categorical(logits=logits_with_i)
    # with_i_prob = with_i_distri.probs

    # # 计算KL散度来评估依赖性
    # kl_divergence_i_given_j = torch.sum(torch.where(original_prob != 0, torch.log(original_prob / with_j_prob), 0), dim=-1)
    # kl_divergence_j_given_i = torch.sum(torch.where(with_i_prob != 0, torch.log(with_i_prob / original_prob), 0), dim=-1)

    # # 如果j对i的影响大于i对j的影响，则i依赖于j
    # return (kl_divergence_i_given_j > kl_divergence_j_given_i)

# # 修改 discrete_autoregreesive_act 函数以使用批处理
# def discrete_autoregreesive_act(args, decoder, obs_rep, obs, batch_size, n_agent, action_dim, tpdv,
                                # available_actions=None, deterministic=False):
    # shifted_action = torch.zeros((obs_rep.size(0), n_agent, action_dim + 1)).to(**tpdv)
    # shifted_action[:, 0, 0] = 1
    # output_action = torch.zeros((obs_rep.size(0), n_agent, 1), dtype=torch.long).to(**tpdv)
    # output_action_log = torch.zeros_like(output_action, dtype=torch.float32).to(**tpdv)
    
    # # 学习较优顺序
    # perm = torch.arange(n_agent).long()
    # new_perm = torch.arange(n_agent).long()
    
    # # 创建 n * k 个随机 i-j 对
    # k=args.k_steps
    # pairs = []
    # for i in range(n_agent):
        # # 随机选择 k 个不同于 i 的 j
        # possible_js = [j for j in range(n_agent) if j != i]
        # selected_js = torch.tensor(possible_js)[torch.randperm(len(possible_js))[:k]]
        # pairs.extend([(i, j.item()) for j in selected_js])
    # pairs = torch.tensor(pairs, dtype=torch.long)
    
    # # 批量检查依赖关系
    # dependencies = check_dependency_batch(decoder, obs_rep, obs, available_actions, pairs, action_dim, tpdv, shifted_action)

    
    # # 根据依赖关系更新排列
    # for idx, (i, j) in enumerate(pairs):
        # if dependencies[0, idx]:# 使用第一个 batch 的结果,dependencies:[batch_size,n_agent.pairs]
            # new_perm[[i, j]] = new_perm[[j, i]]
    
    # perm = new_perm.clone()  # 更新顺序
    # # 使用改正的新的顺序进行决策
    # for i in range(n_agent):
        # logit = decoder(shifted_action, obs_rep, obs)[:, perm[i], :]
        # if available_actions is not None:
            # logit[available_actions[:, perm[i], :] == 0] = -1e10
        # distri = Categorical(logits=logit)
        # action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
        # action_log = distri.log_prob(action)

        # output_action[:, perm[i], :] = action.unsqueeze(-1)
        # output_action_log[:, perm[i], :] = action_log.unsqueeze(-1)
        # if i + 1 < n_agent:
            # shifted_action[:, i + 1, 1:] = F.one_hot(action, num_classes=action_dim)

    # return output_action, output_action_log  


def discrete_parallel_act(decoder, obs_rep, obs, action, batch_size, n_agent, action_dim, tpdv,
                          available_actions=None):
    one_hot_action = F.one_hot(action.squeeze(-1), num_classes=action_dim)  # (batch, n_agent, action_dim)
    shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1)).to(**tpdv)
    shifted_action[:, 0, 0] = 1
    shifted_action[:, 1:, 1:] = one_hot_action[:, :-1, :]
    logit = decoder(shifted_action, obs_rep, obs)
    if available_actions is not None:
        logit[available_actions == 0] = -1e10

    distri = Categorical(logits=logit)
    action_log = distri.log_prob(action.squeeze(-1)).unsqueeze(-1)
    entropy = distri.entropy().unsqueeze(-1)
    return action_log, entropy


def continuous_autoregreesive_act(decoder, obs_rep, obs, batch_size, n_agent, action_dim, tpdv,
                                  deterministic=False):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim)).to(**tpdv)
    output_action = torch.zeros((batch_size, n_agent, action_dim), dtype=torch.float32)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)

    for i in range(n_agent):
        act_mean = decoder(shifted_action, obs_rep, obs)[:, i, :]
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


def continuous_parallel_act(decoder, obs_rep, obs, action, batch_size, n_agent, action_dim, tpdv):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim)).to(**tpdv)
    shifted_action[:, 1:, :] = action[:, :-1, :]

    act_mean = decoder(shifted_action, obs_rep, obs)
    action_std = torch.sigmoid(decoder.log_std) * 0.5
    distri = Normal(act_mean, action_std)

    # log_std = torch.zeros_like(act_mean).to(**tpdv) + decoder.log_std
    # distri = Normal(act_mean, log_std.exp())

    action_log = distri.log_prob(action)
    entropy = distri.entropy()
    return action_log, entropy
