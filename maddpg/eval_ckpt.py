from utils import make_env, dict2csv
import numpy as np
import contextlib
import torch
import os
import time
import argparse
import random
from utils import copy_actor_policy

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def eval_model_q(args, agent):
    eval_env = make_env(args.scenario, args)
    eval_env.seed(args.seed + 10)
    eval_rewards = []
    with temp_seed(args.seed):
        for n_eval in range(args.num_eval_runs):
            obs_n = eval_env.reset()
            episode_reward = 0
            episode_step = 0
            n_agents = eval_env.n
            agents_rew = [[] for _ in range(n_agents)]
            while True:
                action_n = agent.select_action(torch.Tensor(obs_n), action_noise=True,
                                               param_noise=False).squeeze().cpu().numpy()
                next_obs_n, reward_n, done_n, _ = eval_env.step(action_n)
                episode_step += 1
                terminal = (episode_step >= args.num_steps)
                episode_reward += np.sum(reward_n)
                for i, r in enumerate(reward_n):
                    agents_rew[i].append(r)
                obs_n = next_obs_n
                if done_n[0] or terminal:
                    eval_rewards.append(episode_reward)
                    if n_eval % 100 == 0:
                        print('test reward', episode_reward)
                    break

        print("========================================================")
        print(eval_rewards)
        print("{} eval runs".format(args.num_eval_runs,))
        print("GOOD reward: avg {} std {}".format(np.mean(eval_rewards), np.std(eval_rewards)))
        eval_env.close()


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--scenario', required=True,
                        help='name of the environment to run')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed (default: 4)')
    parser.add_argument('--num_steps', type=int, default=25, metavar='N',
                        help='max episode length (default: 1000)')
    parser.add_argument('--num_eval_runs', type=int, default=400, help='number of runs per evaluation (default: 5)')
    parser.add_argument('--pretrained_model', default=None, type=str)
    parser.add_argument('--vis_range', type=float, default=1.6)
    args = parser.parse_args()

    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.pretrained_model:
        agent_pretrained = torch.load(args.pretrained_model)['agents']
        print("Loaded ckpt from " + args.pretrained_model)
    agent = agent_pretrained
    copy_actor_policy(agent, agent)
    eval_model_q(args, agent)