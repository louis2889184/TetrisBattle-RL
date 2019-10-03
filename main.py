import copy
import glob
import os
import time
import random
from collections import deque

from PIL import Image

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate

def exploration_rate(now_game, method='exp'):
    eps_start = 1.0
    eps_end = 0.05
    eps_decay = 2000
    if method == 'exp':

        return eps_end + (eps_start - eps_end) * np.exp(-1.0 * float(now_game) / eps_decay)
    elif method == 'tanh':
        offset = 3.0 * eps_decay
        return eps_end + (eps_start - eps_end) * \
                (1.0 - np.tanh((float(now_game) - offset) / eps_decay)) / 2.0


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    save_name = '%s_%s' % (args.env_name, args.algo)
    if args.postfix != '':
        save_name += ('_' + args.postfix)

    logger_filename = os.path.join(log_dir, save_name)
    logger = utils.create_logger(logger_filename)

    torch.set_num_threads(1)
    device = torch.device("cuda:%d" % args.gpu if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False, 4, 
                         obs_type="grid" if args.grid else "image", skip_frames=args.num_skip_frames)

    if args.load_dir != None:
        actor_critic, ob_rms = \
                torch.load(os.path.join(args.load_dir), map_location=lambda storage, loc: storage)
        vec_norm = utils.get_vec_normalize(envs)
        if vec_norm is not None:
            vec_norm.ob_rms = ob_rms
        print("load pretrained...")
    else:
        actor_critic = Policy(
            envs.observation_space.shape,
            envs.action_space,
            base="grid" if args.grid else None,
            base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))

        gail_train_loader = torch.utils.data.DataLoader(
            gail.ExpertDataset(
                file_name, num_trajectories=4, subsample_frequency=20),
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    lines = deque(maxlen=10)
    start = time.time()
    kk = 0
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    # learning_start = 1000
    learning_start = 0
    best_reward = -100
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        explore = exploration_rate(j - learning_start, 'exp')
        # print(j)
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # if j < learning_start:
            #     action[0, 0] = random.randint(0, envs.action_space.n - 1)
            # elif random.uniform(0, 1) < explore:
            #     action[0, 0] = random.randint(0, envs.action_space.n - 1)
            # else:
            #     pass

            # Obser reward and next obs
            # action[0, 0] = 1
            # envs.take_turns()
            obs, reward, done, infos = envs.step(action)
            # print(obs)
            
            # im = Image.fromarray(obs[0].reshape(224 * 4, -1).cpu().numpy().astype(np.uint8))
            # im.save("samples/%d.png" % kk)
            # info = infos[0]
            # if len(info) > 0:
            #     print(info)
            # print(done)
            # print(infos)
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                if 'sent' in info.keys():
                    lines.append(info['sent'])
            
            # kk += 1
            # print(action.shape)
            # print(obs.shape)
            # print(done.shape)
            # if done[0]:
            #     print(time.time() - start)
            #     print(kk)
            #     exit()

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "" \
                and np.mean(episode_rewards) > best_reward:
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass
            best_reward = np.mean(episode_rewards)
            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, save_name + ".pt"))

        # print(episode_rewards)

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            if j < learning_start:
                logger.info("random action")
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            logger.info(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

            logger.info(
                ' lines sent: mean/median lines {:.1f}/{:.1f}, min/max lines {:.1f}/{:.1f}\n'
                .format(np.mean(lines), np.median(lines), 
                    np.min(lines), np.max(lines)))
            
        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
    main()
