#!/usr/bin/env python
import sys
import os
#import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
import time
sys.path.append("../../")
from ADAPT.config import get_config
from ADAPT.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv

from ADAPT.runner.shared.smac_runner import SMACRunner as Runner

"""Train script for SMAC."""

def parse_smacv2_distribution(args):
    units = args.units.split('v')
    distribution_config = {
        "n_units": int(units[0]),
        "n_enemies": int(units[1]),
        "start_positions": {
            "dist_type": "surrounded_and_reflect",
            "p": 0.5,
            "map_x": 32,
            "map_y": 32,
        }
    }
    if 'protoss' in args.map_name:
        distribution_config['team_gen'] = {
            "dist_type": "weighted_teams",
            "unit_types": ["stalker", "zealot", "colossus"],
            "weights": [0.45, 0.45, 0.1],
            "observe": True,
        }
    elif 'zerg' in args.map_name:
        distribution_config['team_gen'] = {
            "dist_type": "weighted_teams",
            "unit_types": ["zergling", "baneling", "hydralisk"],
            "weights": [0.45, 0.1, 0.45],
            "observe": True,
        } 
    elif 'terran' in args.map_name:
        distribution_config['team_gen'] = {
            "dist_type": "weighted_teams",
            "unit_types": ["marine", "marauder", "medivac"],
            "weights": [0.45, 0.45, 0.1],
            "observe": True,
        } 
    return distribution_config

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2":
                from ADAPT.envs.starcraft2.StarCraft2_Env import StarCraft2Env
                env = StarCraft2Env(all_args)
            elif all_args.env_name == "StarCraft2v2":
                from ADAPT.envs.starcraft2.SMACv2_modified import SMACv2
                env = SMACv2(capability_config=parse_smacv2_distribution(all_args), map_name=all_args.map_name)
            elif all_args.env_name == "SMAC":
                from ADAPT.envs.starcraft2.SMAC import SMAC
                env = SMAC(map_name=all_args.map_name)
            elif all_args.env_name == "SMACv2":
                from ADAPT.envs.starcraft2.SMACv2 import SMACv2
                env = SMACv2(capability_config=parse_smacv2_distribution(all_args), map_name=all_args.map_name)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(1 + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2":
                from ADAPT.envs.starcraft2.StarCraft2_Env import StarCraft2Env
                env = StarCraft2Env(all_args)
            elif all_args.env_name == "StarCraft2v2":
                from ADAPT.envs.starcraft2.SMACv2_modified import SMACv2
                env = SMACv2(capability_config=parse_smacv2_distribution(all_args), map_name=all_args.map_name)
            elif all_args.env_name == "SMAC":
                from ADAPT.envs.starcraft2.SMAC import SMAC
                env = SMAC(map_name=all_args.map_name)
            elif all_args.env_name == "SMACv2":
                from ADAPT.envs.starcraft2.SMACv2 import SMACv2
                env = SMACv2(capability_config=parse_smacv2_distribution(all_args), map_name=all_args.map_name)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--map_name', type=str, default='3m', help="Which smac map to run on")
    parser.add_argument('--eval_map_name', type=str, default='3m', help="Which smac map to eval on")
    parser.add_argument('--run_dir', type=str, default='', help="Which smac map to eval on")
    parser.add_argument("--add_move_state", action='store_true', default=False)
    parser.add_argument("--add_local_obs", action='store_true', default=False)
    parser.add_argument("--add_distance_state", action='store_true', default=False)
    parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
    parser.add_argument("--add_agent_id", action='store_true', default=False)
    parser.add_argument("--add_visible_state", action='store_true', default=False)
    parser.add_argument("--add_xy_state", action='store_true', default=False)
    parser.add_argument("--use_state_agent", action='store_false', default=True)
    parser.add_argument("--use_mustalive", action='store_false', default=True)
    parser.add_argument("--add_center_xy", action='store_false', default=True)
    parser.add_argument("--random_agent_order", action='store_true', default=False)
    parser.add_argument("--sight_range", type=int, default=9)
    parser.add_argument("--shoot_range", type=int, default=6)
    parser.add_argument('--units', type=str, default='10v10') # for smac v2

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if "dec" in all_args.algorithm_name:
        all_args.dec_actor = True
        # all_args.share_actor = False

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args.env_name / all_args.map_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                              str(all_args.experiment_name) +
                              "_seed" + str(all_args.seed),
                         group=all_args.map_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        import time
        timestr = time.strftime("%y%m%d-%H%M%S")
        curr_run = all_args.prefix_name + "-" + timestr
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

    # seed
    torch.manual_seed(all_args.alg_seed)
    torch.cuda.manual_seed_all(all_args.alg_seed)
    np.random.seed(all_args.alg_seed)

    # env
    all_args.run_dir = run_dir
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    if all_args.env_name == "SMAC":
        from smac.env.starcraft2.maps import get_map_params
        num_agents = get_map_params(all_args.map_name)["n_agents"]
    elif all_args.env_name == 'StarCraft2':
        from ADAPT.envs.starcraft2.smac_maps import get_map_params
        num_agents = get_map_params(all_args.map_name)["n_agents"]
    elif all_args.env_name == "SMACv2" or all_args.env_name == 'StarCraft2v2':
        from smacv2.env.starcraft2.maps import get_map_params
        num_agents = parse_smacv2_distribution(all_args)['n_units']
        
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
