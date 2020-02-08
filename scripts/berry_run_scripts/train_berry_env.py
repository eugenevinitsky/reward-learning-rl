import argparse
from copy import deepcopy

import ray
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as DEFAULT_PPO_CONFIG
from ray.tune import run as run_tune
from ray.tune.registry import register_env

from envs.berry_env import BerryEnv

def ray_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_title', type=str, default='test',
                        help='Informative experiment title to help distinguish results')
    parser.add_argument('--num_cpus', type=int, default=1, help='Number of cpus to run experiment with')
    parser.add_argument('--local_mode', action='store_true', help='Set to true if this will '
                                                                  'be run in local mode')
    parser.add_argument('--train_batch_size', type=int, default=100, help='How many steps go into a training batch')
    parser.add_argument('--num_iters', type=int, default=350)
    parser.add_argument('--checkpoint_freq', type=int, default=50)
    parser.add_argument('--render', type=str, default=False)

    return parser


def env_creator(config):
    return BerryEnv()


def setup_exps():
    parser = ray_parser()
    args = parser.parse_args()

    config = deepcopy(DEFAULT_PPO_CONFIG)
    config['seed'] = 0
    config['train_batch_size'] = args.train_batch_size
    config['gamma'] = 0.995

    # Universal hyperparams
    config['num_workers'] = args.num_cpus
    config["batch_mode"] = "complete_episodes"

    # config['num_adversaries'] = args.num_adv
    # config['kl_diff_weight'] = args.kl_diff_weight
    # config['kl_diff_target'] = args.kl_diff_target
    # config['kl_diff_clip'] = 5.0

    config['env'] = "BerryEnv"
    register_env("BerryEnv", env_creator)

    # create a custom string that makes looking at the experiment names easier
    def trial_str_creator(trial):
        return "{}_{}".format(trial.trainable_name, trial.experiment_tag)

    stop_dict = {}
    stop_dict.update({
        'training_iteration': args.num_iters
    })

    exp_dict = {
        'name': args.exp_title,
        'run_or_experiment': PPOTrainer,
        'trial_name_creator': trial_str_creator,
        # 'checkpoint_freq': args.checkpoint_freq,
        'checkpoint_at_end': True,
        'stop': stop_dict,
        'config': config,
        'num_samples': args.num_samples,
    }
    return exp_dict, args

if __name__ == "__main__":
    exp_dict, args = setup_exps()

    if args.multi_node:
        ray.init(redis_address='localhost:6379')
    elif args.local_mode:
        ray.init(local_mode=True)
    else:
        ray.init()

    run_tune(**exp_dict, queue_trials=False, raise_on_failed_trial=False)
