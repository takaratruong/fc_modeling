from stable_baselines3.common.callbacks import BaseCallback
import torch


class VideoCallback(BaseCallback):
    def __init__(self, vid_env, eval_freq, verbose=0):
        super(VideoCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        self.vid_env = vid_env

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0: # <-- fix later
            for _ in range(4):
                obs = self.vid_env.reset()
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=False)
                    obs, _, done, _ = self.vid_env.step(action)

            self.vid_env.close()
        return True


class AMPVideoCallback:
    def __init__(self, vid_env):
        self.vid_env = vid_env

    def save_video(self, model):
        for _ in range(3):
            obs = self.vid_env.reset()
            done = False
            while not done:
                action = model.sample_best_actions(torch.tensor(obs).float().cuda()).detach().cpu().numpy()
                obs, _, done, _ = self.vid_env.step(action)

        self.vid_env.close()
        return True
