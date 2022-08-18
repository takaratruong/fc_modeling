from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

from environments.walker2d.walker2d_env import WalkerEnv
from environments.walker2d.walker2d_fixed_env import WalkerFixedEnv

from environments.humanoid.humanoid_env import HumanoidEnv
from environments.humanoid.humanoid_treadmill_env import HumanoidTreadmillEnv

from environments.skeleton.skeleton_env import SkeletonEnv
from environments.rajagopal.rajagopal_env import RajagopalEnv


def load(env, args, run):
    train_env = make_vec_env(lambda: env(args=args), n_envs=args.num_envs, vec_env_cls=SubprocVecEnv) if args.policy_path is None else None
    eval_env = make_vec_env(lambda: env(args=args), n_envs=1)
    vid_env = VecVideoRecorder(make_vec_env(lambda: env(args=args), seed=0, n_envs=1), args.log_dir + f"videos/{run.id}", record_video_trigger=lambda x: x == 0) if run is not None else None

    return train_env, eval_env, vid_env

def load_environments(args, run=None):

    assert args.environment in ['walker2d', 'walker2d_fixed', 'humanoid', 'humanoid_treadmill', 'rajagopal', 'skeleton'], 'args.environment not included in loader'

    if args.environment == 'walker2d':
        return load(WalkerEnv, args, run)

    if args.environment == 'walker2d_fixed':
        args.fixed_gait_policy = PPO.load(args.fixed_gait_policy_path) # <- path to model from gait_modeling directory'
        return load(WalkerFixedEnv, args, run)

    if args.environment == 'humanoid':
        return load(HumanoidEnv, args, run)

    if args.environment == 'humanoid_treadmill':
        return load(HumanoidTreadmillEnv, args, run)

    if args.environment == 'rajagopal':
        return load(RajagopalEnv, args, run)

    if args.environment == 'skeleton':
        return load(SkeletonEnv, args, run)


