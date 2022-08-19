import wandb
from configs.config_loader import load_args
from learning_algs.amp_ppo import RL
from utils.video_callback import AMPVideoCallback
from utils.load_envs import load_environments
import utils.messages as message

if __name__ == '__main__':
    args = load_args()
    message.header_message(args)

    # Initialize wandb
    run = None
    if args.wandb:
        run = wandb.init(project=args.project_name, config=args, name=args.exp_name, monitor_gym=True, dir=args.log_dir)
        wandb.define_metric("step", hidden=True)
        wandb.define_metric("eval/reward", step_metric="step")
        wandb.define_metric("eval/ep_len", step_metric="step")

        wandb.define_metric("train/critic loss", step_metric="step")
        wandb.define_metric("train/actor loss", step_metric="step")
        wandb.define_metric("train/disc loss", step_metric="step")

    # Load environments
    train_env, _, vid_env = load_environments(args, run)
    assert train_env is not None, 'Train env is None'

    # Create callbacks
    vid_callback = AMPVideoCallback(vid_env) if vid_env is not None else None

    # Instantiate algorithm
    ppo = RL(train_env, vid_callback, args)

    # Train
    ppo.collect_samples_multithread()





