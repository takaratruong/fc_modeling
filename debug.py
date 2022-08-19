from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from gym.wrappers import RecordVideo
import gym
from environments.humanoid.humanoid_treadmill_env import HumanoidTreadmillEnv
from environments.humanoid.humanoid_v4 import HumanoidEnv
from configs.config_loader import load_args
from utils.video_callback import AMPVideoCallback


if __name__ == '__main__':
    # args = load_args()
    # env = HumanoidTreadmillEnv(args=args)
    #
    # vid_env = VecVideoRecorder(make_vec_env(lambda: HumanoidTreadmillEnv(args=args), n_envs=1), args.log_dir, record_video_trigger=lambda x: x == 0)
    #
    # vid_callback = AMPVideoCallback(vid_env)
    #

    env = gym.make("Humanoid-v4", render_mode='human')

    env.reset()

    for _ in range(1000):
        env.step(env.action_space.sample())

    frame_collection = env.render()
    env.close()


