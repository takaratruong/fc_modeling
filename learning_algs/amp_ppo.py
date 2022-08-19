import torch
import numpy as np
import time
import torch.multiprocessing as mp
import pickle
import torch.optim as optim
import wandb
import os
from scipy.interpolate import interp1d
import ipdb

from learning_algs.amp_models import ActorCriticNet, Discriminator
from mocap.mocap import MoCap
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPOStorage:
    def __init__(self, num_inputs, num_outputs, max_size=64000):
        self.states = torch.zeros(max_size, num_inputs).to(device)
        self.next_states = torch.zeros(max_size, num_inputs).to(device)
        self.actions = torch.zeros(max_size, num_outputs).to(device)
        self.dones = torch.zeros(max_size, 1, dtype=torch.int8).to(device)
        self.log_probs = torch.zeros(max_size).to(device)
        self.rewards = torch.zeros(max_size).to(device)
        self.q_values = torch.zeros(max_size, 1).to(device)
        self.mean_actions = torch.zeros(max_size, num_outputs).to(device)
        self.counter = 0
        self.sample_counter = 0
        self.max_samples = max_size
    def sample(self, batch_size):
        idx = torch.randint(self.counter, (batch_size,), device=device)
        return self.states[idx, :], self.actions[idx, :], self.next_states[idx, :], self.rewards[idx], self.q_values[idx, :], self.log_probs[idx]
    def clear(self):
        self.counter = 0
    def push(self, states, actions, next_states, rewards, q_values, log_probs, size):
        self.states[self.counter:self.counter + size, :] = states.detach().clone()
        self.actions[self.counter:self.counter + size, :] = actions.detach().clone()
        self.next_states[self.counter:self.counter + size, :] = next_states.detach().clone()
        self.rewards[self.counter:self.counter + size] = rewards.detach().clone()
        self.q_values[self.counter:self.counter + size, :] = q_values.detach().clone()
        self.log_probs[self.counter:self.counter + size] = log_probs.detach().clone()
        self.counter += size

    def discriminator_sample(self, batch_size):
        if self.sample_counter == 0 or self.sample_counter == self.max_samples:
            self.permute()
        self.sample_counter %= self.max_samples
        self.sample_counter += batch_size

        return self.states[self.sample_counter - batch_size:self.sample_counter, :], self.next_states[self.sample_counter - batch_size:self.sample_counter, :]

    def critic_sample(self, batch_size):
        if self.sample_counter == 0 or self.sample_counter == self.max_samples:
            self.permute()
        self.sample_counter %= self.max_samples
        self.sample_counter += batch_size
        return self.states[self.sample_counter - batch_size:self.sample_counter, :], self.q_values[self.sample_counter - batch_size:self.sample_counter,:]

    def actor_sample(self, batch_size):
        if self.sample_counter == 0 or self.sample_counter == self.max_samples:
            self.permute()
        self.sample_counter %= self.max_samples
        self.sample_counter += batch_size
        return self.states[self.sample_counter - batch_size:self.sample_counter, :], self.actions[self.sample_counter - batch_size:self.sample_counter, :], self.q_values[self.sample_counter - batch_size:self.sample_counter, :], self.log_probs[self.sample_counter - batch_size:self.sample_counter]

    def permute(self):
        permuted_index = torch.randperm(self.max_samples)
        self.states[:, :] = self.states[permuted_index, :]
        self.actions[:, :] = self.actions[permuted_index, :]
        self.q_values[:, :] = self.q_values[permuted_index, :]
        self.log_probs[:] = self.log_probs[permuted_index]


class RL(object):
    def __init__(self, env, vid_callback, args):

        self.env = env
        self.params = args
        self.vid_callback = vid_callback
        # self.env.env.disableViewer = False
        self.num_inputs = env.observation_space.shape[0]
        self.num_outputs = env.action_space.shape[0]
        self.hidden_layer = [args.model_hidden_layer_size, args.model_hidden_layer_size] #if hidden_layer is None else hidden_layer

        #self.params = Params()
        self.Net = ActorCriticNet

        self.model = self.Net(self.num_inputs, self.num_outputs, self.hidden_layer)

        obs_size = env.get_obs().shape[1]
        # self.discriminator = Discriminator((obs_size-1-3)*2, [args.disc_hidden_layer_size, args.disc_hidden_layer_size]) # HARD CODED: -1 for phase *2 for d

        self.discriminator = Discriminator((obs_size-1)*2, [args.disc_hidden_layer_size, args.disc_hidden_layer_size]) # HARD CODED: -1 for phase *2 for d
        self.model.share_memory()
        self.test_mean = []
        self.test_std = []

        self.noisy_test_mean = []
        self.noisy_test_std = []
        self.lr = self.params.lr

        self.test_list = []
        self.noisy_test_list = []

        self.best_score_queue = mp.Queue()
        self.best_score = mp.Value("f", 0)
        self.max_reward = mp.Value("f", 1)

        self.best_validation = 1.0
        self.current_best_validation = 1.0

        self.gpu_model = self.Net(self.num_inputs, self.num_outputs, self.hidden_layer)
        self.gpu_model.to(device)
        self.model_old = self.Net(self.num_inputs, self.num_outputs, self.hidden_layer).to(device)
        self.discriminator.to(device)

        self.base_controller = None
        self.base_policy = None

        self.total_rewards = []
        self.episode_lengths = []

        self.actor_optimizer = optim.AdamW(self.gpu_model.parameters(), lr=1e-4)
        self.critic_optimizer = optim.AdamW(self.gpu_model.parameters(), lr=1e-4)

    def run_test(self, num_test=1):
        state = self.env.reset()
        ave_test_reward = 0

        total_rewards = []
        if self.num_envs > 1:
            test_index = 1
        else:
            test_index = 0

        for i in range(num_test):
            total_reward = 0
            while True:
                state = self.shared_obs_stats.normalize(state)
                mu = self.gpu_model.sample_best_actions(state)
                state, reward, done, _ = self.env.step(mu)
                total_reward += reward[test_index].item()

                if done[test_index]:
                    state = self.env.reset()
                    # print(self.env.position)
                    # print(self.env.time)
                    ave_test_reward += total_reward / num_test
                    total_rewards.append(total_reward)
                    break

        # print("avg test reward is", ave_test_reward)
        reward_mean = np.mean(total_rewards)
        reward_std = np.std(total_rewards)
        self.test_mean.append(reward_mean)
        self.test_std.append(reward_std)
        self.test_list.append((reward_mean, reward_std))
        # print(self.model.state_dict())

    def run_test_with_noise(self, num_test=10):

        reward_mean = np.mean(self.total_rewards)
        reward_std = np.std(self.total_rewards)

        ep_len_mean = np.mean(self.episode_lengths)
        ep_len_std = np.std(self.episode_lengths)

        # print(reward_mean, reward_std, self.total_rewards)
        self.noisy_test_mean.append(reward_mean)
        self.noisy_test_std.append(reward_std)
        self.noisy_test_list.append((reward_mean, reward_std))

        # print("reward mean,", reward_mean)
        # print("reward std,", reward_std)

        return reward_mean, reward_std , ep_len_mean, ep_len_std

    def feature_extractor(self, state):
        return state

    def collect_samples_vec(self, num_samples, start_state=None, noise=-2.5, env_index=0, random_seed=1):

        start_state = self.env.get_obs()

        samples = 0
        done = False
        states = []
        next_states = []
        actions = []
        # mean_actions = []
        rewards = []
        values = []
        q_values = []
        real_rewards = []
        log_probs = []
        dones = []
        noise = self.base_noise * self.explore_noise.value
        self.gpu_model.set_noise(noise)

        state = start_state

        total_reward1 = 0
        total_reward2 = 0
        calculate_done1 = False
        calculate_done2 = False
        self.total_rewards = []
        start = time.time()
        state = torch.from_numpy(state).to(device).type(torch.cuda.FloatTensor) #DIFF
        while samples < num_samples:
            with torch.no_grad():
                action, mean_action = self.gpu_model.sample_actions(state)
                log_prob = self.gpu_model.calculate_prob(state, action, mean_action)

            states.append(state.clone())
            actions.append(action.clone())
            log_probs.append(log_prob.clone())
            next_state, reward, done, _ = self.env.step(action.cpu().numpy()) #< ----- change

            # rewards.append(reward.clone())
            next_state = torch.from_numpy(next_state).to(device).type(torch.cuda.FloatTensor)
            reward = torch.from_numpy(reward).to(device).type(torch.cuda.FloatTensor)
            done = torch.from_numpy(done).to(device).type(torch.cuda.IntTensor)

            dones.append(done.clone())

            next_states.append(next_state.clone())

            if self.params.alg == 'amp':
                reward = self.discriminator.compute_disc_reward(state[:,:-1], next_state[:,:-1]) # DISC
            rewards.append(reward.clone())

            state = next_state.clone()

            samples += 1

            self.env.reset_time_limit()
        #print("sim time", time.time() - start)
        start = time.time()
        counter = num_samples - 1
        R = self.gpu_model.get_value(state)
        while counter >= 0:
            R = R * (1 - dones[counter].unsqueeze(-1))
            R = 0.99 * R + rewards[counter].unsqueeze(-1)
            q_values.insert(0, R)
            counter -= 1
            # print(len(q_values))

        for i in range(num_samples):
            self.storage.push(states[i], actions[i], next_states[i], rewards[i], q_values[i], log_probs[i], self.num_envs)

        self.total_rewards = self.env.get_total_reward()

        self.episode_lengths = self.env.get_elapsed_time()
        #print("processing time", time.time() - start)


    def sample_expert_motion(self, batch_size):

        states, next_states = self.mocap.sample_expert(batch_size)

        return torch.from_numpy(states).float().to(device), torch.from_numpy(next_states).float().to(device)

    """ FIX """
    def update_discriminator(self, batch_size, num_epoch):
        self.discriminator.train()
        optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr)
        epoch_loss = 0

        for k in range(num_epoch):
            batch_states, batch_next_states = self.storage.discriminator_sample(batch_size)

            policy_d = self.discriminator.compute_disc(batch_states[:,:-1], batch_next_states[:,:-1]) # # DISC
            policy_loss = (policy_d + torch.ones(policy_d.size(), device=device)) ** 2
            policy_loss = policy_loss.mean()

            batch_expert_states, batch_expert_next_states = self.sample_expert_motion(batch_size)

            expert_d = self.discriminator.compute_disc(batch_expert_states, batch_expert_next_states)
            expert_loss = (expert_d - torch.ones(expert_d.size(), device=device)) ** 2
            expert_loss = expert_loss.mean()

            grad_penalty = self.discriminator.grad_penalty(batch_expert_states, batch_expert_next_states)

            total_loss = policy_loss + expert_loss + 5 * grad_penalty
            epoch_loss += total_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        return epoch_loss/num_epoch

    # removed fresh update
    def update_critic(self, batch_size, num_epoch):
        self.gpu_model.train()
        optimizer = self.critic_optimizer  #optim.Adam(self.gpu_model.parameters(), lr=10 * self.lr)

        storage = self.storage
        gpu_model = self.gpu_model
        epoch_loss = 0

        for k in range(num_epoch):
            batch_states, batch_q_values = storage.critic_sample(batch_size)
            batch_q_values = batch_q_values  # / self.max_reward.value
            v_pred = gpu_model.get_value(batch_states)

            loss_value = (v_pred - batch_q_values) ** 2
            loss_value = 0.5 * loss_value.mean()
            epoch_loss += loss_value

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        return epoch_loss/num_epoch


    def update_actor(self, batch_size, num_epoch):
        self.gpu_model.train()
        optimizer = self.actor_optimizer #optim.Adam(self.gpu_model.parameters(), lr=self.lr)

        storage = self.storage
        gpu_model = self.gpu_model
        model_old = self.model_old
        params_clip = self.params.clip

        epoch_loss = 0

        for k in range(num_epoch):
            batch_states, batch_actions, batch_q_values, batch_log_probs = storage.actor_sample(batch_size)

            batch_q_values = batch_q_values  # / self.max_reward.value

            with torch.no_grad():
                v_pred_old = gpu_model.get_value(batch_states)

            batch_advantages = (batch_q_values - v_pred_old)

            probs, mean_actions = gpu_model.calculate_prob_gpu(batch_states, batch_actions)
            probs_old = batch_log_probs  # model_old.calculate_prob_gpu(batch_states, batch_actions)
            ratio = (probs - (probs_old)).exp()
            ratio = ratio.unsqueeze(1)
            surr1 = ratio * batch_advantages
            surr2 = ratio.clamp(1 - params_clip, 1 + params_clip) * batch_advantages
            loss_clip = -(torch.min(surr1, surr2)).mean()

            total_loss = loss_clip + 0.001 * (mean_actions ** 2).mean()
            epoch_loss += total_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        # print(self.shared_obs_stats.mean.data)
        if self.lr > 1e-4:
            self.lr *= 0.99
        else:
            self.lr = 1e-4

        return epoch_loss/num_epoch

    def save_model(self, filename):
        torch.save(self.gpu_model.state_dict(), filename)

    def save_shared_obs_stas(self, filename):
        with open(filename, 'wb') as output:
            pickle.dump(self.shared_obs_stats, output, pickle.HIGHEST_PROTOCOL)

    def save_statistics(self, filename):
        statistics = [self.time_passed, self.num_samples, self.test_mean, self.test_std, self.noisy_test_mean, self.noisy_test_std]
        with open(filename, 'wb') as output:
            pickle.dump(statistics, output, pickle.HIGHEST_PROTOCOL)

    def collect_samples_multithread(self):
        # queue = Queue.Queue()
        self.num_envs = self.params.num_envs
        self.start = time.time()
        self.lr = 1e-3
        self.weight = 10
        num_threads = 1
        self.num_samples = 0
        self.time_passed = 0
        score_counter = 0
        total_thread = 0
        max_samples = self.num_envs*self.params.num_steps #6000

        self.storage = PPOStorage(self.num_inputs, self.num_outputs, max_size=max_samples)
        seeds = [i * 100 for i in range(num_threads)]

        self.explore_noise = mp.Value("f", -2.0)
        self.base_noise = np.ones(self.num_outputs)
        noise = self.base_noise * self.explore_noise.value
        self.model.set_noise(noise)
        self.gpu_model.set_noise(noise)
        self.env.reset()

        # train_path = '/home/takaraet/gait_modeling/mocap/mocap_data/skeleton/Subject2_exp'
        # val_path = '/home/takaraet/gait_modeling/mocap/mocap_data/skeleton/Subject2_exp'
        # self.mocap = MoCap(train_path, val_path, self.params.frame_skip, .01)

        for iterations in range(200000): #200000

            print("------------------------------")
            print("iteration: ", iterations)
            iteration_start = time.time()
            while self.storage.counter < max_samples:
                self.collect_samples_vec(self.params.num_steps//2, noise=noise)
            start = time.time()

            critic_loss = self.update_critic(max_samples // 4, 40)
            actor_loss = self.update_actor(max_samples // 4, 40)

            if self.params.alg == 'amp':
                disc_loss = self.update_discriminator(max_samples // 4, 40)
                if self.params.wandb:
                    wandb.log({"step": iterations, "train/disc loss": disc_loss})

            self.storage.clear()

            if (iterations) % self.params.vid_freq == 0 and self.vid_callback is not None:
                self.vid_callback.save_video(self.gpu_model)

            if (iterations) % 5 == 0:
                reward_mean, reward_std, ep_len_mean, ep_len_std = self.run_test_with_noise(num_test=2)
                print("reward: ", np.round(reward_mean, 3), u"\u00B1" ,np.round(reward_std,3))
                print("ep len: ", np.round(ep_len_mean,3), u"\u00B1" ,np.round(ep_len_std,3))

                if self.params.wandb:
                    wandb.log({"step": iterations, "eval/reward": reward_mean, "eval/ep_len": ep_len_mean} )

            if self.params.wandb:
                wandb.log({"step": iterations, "train/critic loss": critic_loss, "train/actor loss": actor_loss})

            print("iteration time", np.round(time.time() - iteration_start, 3))
            print()

            if (iterations) % 100 == 0:
                best_model_path = self.params.log_dir + 'models/' + self.params.exp_name
                os.makedirs(best_model_path, exist_ok=True)
                torch.save(self.gpu_model.state_dict(), best_model_path +'/' + self.params.exp_name + "_iter%d.pt" % (iterations))

        self.save_reward_stats("reward_stats.npy")
        self.save_model(self.params.log_dir + 'models/' + self.params.exp_name + '/' + self.params.exp_name + "_final.pt" )

    def add_env(self, env):
        self.env_list.append(env)