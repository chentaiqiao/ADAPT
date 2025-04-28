#!/bin/sh
env="StarCraft2v2"
algo="ADAPT"
map="10gen_terran"
exp="single"
seed=1
name="ADAPT"


units="3v3"
exp=units
ppo_epochs=5
ppo_clip=0.05
steps=10000000
epsilon=0.1
echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}, cuda is $1, unit is ${units}"
CUDA_VISIBLE_DEVICES=$1 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
  --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 \
  --episode_length 400 --num_env_steps ${steps} --lr 5e-4 --ppo_epoch ${ppo_epochs} --clip_param ${ppo_clip} --save_interval 100000 \
  --use_value_active_masks --use_eval --prefix_name ${name} --use_recurrent_policy 

# units="10v10"
# echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}, cuda is $1, unit is ${units}"
# CUDA_VISIBLE_DEVICES=$1 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
#   --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 \
#   --episode_length 400 --num_env_steps ${steps} --lr 5e-4 --ppo_epoch ${ppo_epochs} --clip_param ${ppo_clip} --save_interval 100000 \
#   --use_value_active_masks --use_eval --prefix_name ${name} --use_recurrent_policy 

# units="10v11"
# echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}, cuda is $1, unit is ${units}"
# CUDA_VISIBLE_DEVICES=$1 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
#   --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 \
#   --episode_length 400 --num_env_steps ${steps} --lr 5e-4 --ppo_epoch ${ppo_epochs} --clip_param ${ppo_clip} --save_interval 100000 \
#   --use_value_active_masks --use_eval --prefix_name ${name} --use_recurrent_policy 

# units="20v20"
# echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}, cuda is $1, unit is ${units}"
# CUDA_VISIBLE_DEVICES=$1 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
#   --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 \
#   --episode_length 400 --num_env_steps ${steps} --lr 5e-4 --ppo_epoch ${ppo_epochs} --clip_param ${ppo_clip} --save_interval 100000 \
#   --use_value_active_masks --use_eval --prefix_name ${name} --use_recurrent_policy 

# map="10gen_protoss"
# units="5v5"
# echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}, cuda is $1, unit is ${units}"
# CUDA_VISIBLE_DEVICES=$1 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
#   --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 \
#   --episode_length 400 --num_env_steps ${steps} --lr 5e-4 --ppo_epoch ${ppo_epochs} --clip_param ${ppo_clip} --save_interval 100000 \
#   --use_value_active_masks --use_eval --prefix_name ${name} --use_recurrent_policy 

# units="10v10"
# echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}, cuda is $1, unit is ${units}"
# CUDA_VISIBLE_DEVICES=$1 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
#   --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 \
#   --episode_length 400 --num_env_steps ${steps} --lr 5e-4 --ppo_epoch ${ppo_epochs} --clip_param ${ppo_clip} --save_interval 100000 \
#   --use_value_active_masks --use_eval --prefix_name ${name} --use_recurrent_policy 

# units="10v11"
# echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}, cuda is $1, unit is ${units}"
# CUDA_VISIBLE_DEVICES=$1 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
#   --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 \
#   --episode_length 400 --num_env_steps ${steps} --lr 5e-4 --ppo_epoch ${ppo_epochs} --clip_param ${ppo_clip} --save_interval 100000 \
#   --use_value_active_masks --use_eval --prefix_name ${name} --use_recurrent_policy

# units="20v20"
# echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}, cuda is $1, unit is ${units}"
# CUDA_VISIBLE_DEVICES=$1 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
#   --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 \
#   --episode_length 400 --num_env_steps ${steps} --lr 5e-4 --ppo_epoch ${ppo_epochs} --clip_param ${ppo_clip} --save_interval 100000 \
#   --use_value_active_masks --use_eval --prefix_name ${name} --use_recurrent_policy 

# map="10gen_zerg"
# units="5v5"
# echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}, cuda is $1, unit is ${units}"
# CUDA_VISIBLE_DEVICES=$1 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
#   --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 \
#   --episode_length 400 --num_env_steps ${steps} --lr 5e-4 --ppo_epoch ${ppo_epochs} --clip_param ${ppo_clip} --save_interval 100000 \
#   --use_value_active_masks --use_eval --prefix_name ${name} --use_recurrent_policy 

# units="10v10"
# echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}, cuda is $1, unit is ${units}"
# CUDA_VISIBLE_DEVICES=$1 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
#   --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 \
#   --episode_length 400 --num_env_steps ${steps} --lr 5e-4 --ppo_epoch ${ppo_epochs} --clip_param ${ppo_clip} --save_interval 100000 \
#   --use_value_active_masks --use_eval --prefix_name ${name} --use_recurrent_policy 

# units="10v11"
# echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}, cuda is $1, unit is ${units}"
# CUDA_VISIBLE_DEVICES=$1 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
#   --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 \
#   --episode_length 400 --num_env_steps ${steps} --lr 5e-4 --ppo_epoch ${ppo_epochs} --clip_param ${ppo_clip} --save_interval 100000 \
#   --use_value_active_masks --use_eval --prefix_name ${name} --use_recurrent_policy

# units="20v20"
# echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}, cuda is $1, unit is ${units}"
# CUDA_VISIBLE_DEVICES=$1 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
#   --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 \
#   --episode_length 400 --num_env_steps ${steps} --lr 5e-4 --ppo_epoch ${ppo_epochs} --clip_param ${ppo_clip} --save_interval 100000 \
#   --use_value_active_masks --use_eval --prefix_name ${name} --use_recurrent_policy 


