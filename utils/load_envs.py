from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO


from environments.humanoid.humanoid_treadmill_env import HumanoidTreadmillEnv

def load(env, args, run):
    train_env = make_vec_env(lambda: env(args=args), n_envs=args.num_envs, vec_env_cls=SubprocVecEnv) if args.policy_path is None else None
    eval_env = make_vec_env(lambda: env(args=args), n_envs=1)
    vid_env = VecVideoRecorder(make_vec_env(lambda: env(args=args), n_envs=1), args.log_dir + f"videos/{run.id}", record_video_trigger=lambda x: x == 0) if run is not None else None

    return train_env, eval_env, vid_env

def load_environments(args, run=None):

    return load(HumanoidTreadmillEnv, args, run)



