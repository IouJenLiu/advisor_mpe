from utils import make_env, dict2csv
import numpy as np
import contextlib
import torch
from ckpt_plot.plot_curve import plot_result
import os
import time
from torch.utils.tensorboard import SummaryWriter

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def eval_model_q(test_q, done_training, args):
    plot = {'good_rewards': [], 'adversary_rewards': [], 'rewards': [], 'steps': [], 'q_loss': [], 'gcn_q_loss': [],
            'p_loss': [], 'final': [], 'abs': [], 'use_expert_weight': []}
    best_eval_reward = -100000000
    tb_writer = SummaryWriter(log_dir=os.path.join(args.save_dir, args.exp_name))
    while True:
        if not test_q.empty():
            print('=================== start eval ===================')
            eval_env = make_env(args.scenario, args)
            eval_env.seed(args.seed + 10)
            eval_rewards = []
            good_eval_rewards = []
            agent, tr_log = test_q.get()
            with temp_seed(args.seed):
                for n_eval in range(args.num_eval_runs):
                    obs_n = eval_env.reset()
                    episode_reward = 0
                    episode_step = 0
                    n_agents = eval_env.n
                    agents_rew = [[] for _ in range(n_agents)]
                    while True:
                        # action_n = [agent.select_action(torch.Tensor([obs]), action_noise=False, param_noise=False).squeeze().numpy() for obs in obs_n]
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
                            agents_rew = [np.sum(rew) for rew in agents_rew]
                            good_reward = np.sum(agents_rew)
                            good_eval_rewards.append(good_reward)
                            #adversary_eval_rewards.append(adversary_reward)
                            #if n_eval % 100 == 0:
                            #    print('test reward', episode_reward)
                            break
                if np.mean(eval_rewards) > best_eval_reward:
                    best_eval_reward = np.mean(eval_rewards)
                    torch.save({'agents': agent}, os.path.join(tr_log['exp_save_dir'], 'agents_best.ckpt'))

                plot['rewards'].append(np.mean(eval_rewards))
                #plot['good_rewards'].append(np.mean(good_eval_rewards))
                #plot['adversary_rewards'].append(np.mean(adversary_eval_rewards))
                plot['steps'].append(tr_log['total_numsteps'])
                plot['q_loss'].append(tr_log['value_loss'])
                plot['p_loss'].append(tr_log['policy_loss'])



                print("========================================================")
                print(
                    "Episode: {}, total numsteps: {}, {} eval runs, total time: {} s".
                        format(tr_log['i_episode'], tr_log['total_numsteps'], args.num_eval_runs,
                               time.time() - tr_log['start_time']))
                print("GOOD reward: avg {} std {}, average reward: {}, best reward {}".format(np.mean(eval_rewards),
                                                                                              np.std(eval_rewards),
                                                                                              np.mean(plot['rewards'][
                                                                                                      -10:]),
                                                                                              best_eval_reward))
                plot['final'].append(np.mean(plot['rewards'][-10:]))
                plot['abs'].append(best_eval_reward)
                dict2csv(plot, os.path.join(tr_log['exp_save_dir'], 'train_curve.csv'))
                eval_env.close()
                tb_writer.add_scalar('eval/reward', np.mean(eval_rewards), tr_log['total_numsteps'])
                if tr_log['value_loss'] is not None:
                    tb_writer.add_scalar('train/q_loss', tr_log['value_loss'], tr_log['total_numsteps'])
                    tb_writer.add_scalar('train/p_loss', tr_log['policy_loss'], tr_log['total_numsteps'])
                    if 'use_expert_weight' in tr_log:
                        tb_writer.add_scalar('train/use_expert_weight', tr_log['use_expert_weight'], tr_log['total_numsteps'])

        if done_training.value and test_q.empty():
            torch.save({'agents': agent}, os.path.join(tr_log['exp_save_dir'], 'agents.ckpt'))
            break

