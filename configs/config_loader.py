import configargparse
import torch

p = configargparse.ArgParser()

""" Logistics (Naming/Loading/Saving/etc..) """
p.add('-c', '--config', required=True, is_config_file=True, help='config file path')
p.add('-n', '--exp_name', type=str, default='default_exp_name')
p.add('--project_name', type=str, required=True)

p.add('--log_dir', type=str, default='results/')

p.add('--seed', type=int, default=2)
p.add('--wandb',  action='store_true')

p.add('--num_frc', type=int, required=True)
p.add('--hidden_dim', type=int, action='append')
p.add('--num_epochs', type=int, default=1000)
p.add('--batch_size', type=int, default=500)
p.add('--mocap_path', type=str, default=None)

p.add('--percent_train', type=float, default=.85)


p.add('--eval_freq', type=int, default=10)
p.add('--vid_freq', type=int, default=50)

# Gait Parameters
p.add('--gait_ref_file', type=str, default='')
p.add('--actuator_ref_file', type=str, default='')

p.add('--gait_cycle_time', type=float, default=None)
p.add('--gait_cycle_vel', type=float, default=1)
p.add('--treadmill_velocity', type=float, default=1.25)

# Policy Parameters
p.add('--policy_path', type=str, default=None)  # Experiment folder name defining the gait policy (loaded from results)
p.add('--model_hidden_layer_size', type=int, default=128)
p.add('--disc_hidden_layer_size', type=int, default=128)

p.add('--p_gain', type=float, default=1)
p.add('--d_gain', type=float, default=1)

""" Experiment Parameters """
# Visualizing
p.add('-v', '--visualize', action='store_true')  # Visualize policy
p.add('--vis_ref', action='store_true')  # Visualize reference motion

# Training parameters
p.add('-env', '--environment', required=True, type=str)
p.add('--fixed_gait_policy_path', type=str, default='results/models/walk/best_model')

p.add('--num_envs', type=int, default=20)
p.add('--num_steps', type=int, default=2048) # change later for amp
p.add('--time_steps', type=int, default=10e8)

p.add('--frame_skip', type=int, default=10)
p.add('--max_ep_time', type=float, default=10.0)

# Perturbation Parameters
p.add('-rp',  '--rand_perturbation', action='store_true')  # Use random perturbation
p.add('-cp',  '--const_perturbation', action='store_true')  # Use constant perturbation (if neither then no impacts)
p.add('-mp',  '--midstance_perturbation', action='store_true')  # Use random perturbation

p.add('-p_frc', '--perturbation_force', type=float, default=90)
p.add('-p_dir', '--perturbation_dir', type=int, default=1)
p.add('-p_frc_min', '--min_perturbation_force_mag', type=float, default=70)
p.add('-p_dur', '--perturbation_duration', type=float, default=.3)
p.add('-p_del', '--perturbation_delay', type=float, default=3.0)

# Reward/Action weights
p.add('--phase_action_mag', type=float, default=50)

p.add('--jnt_weight', type=float, default=.4)
p.add('--rot_weight', type=float, default=.3)
p.add('--pos_weight', type=float, default=.3)


### FOR AMP ###
p.add('-lr', type=float, default=1e-3)
p.add('-clip', type=float, default=.2)


# not part of configargparse
POLICY_KWARGS = dict(log_std_init=-2.0, ortho_init=True, activation_fn=torch.nn.ReLU,
                     net_arch=[dict(pi=[128, 128], vf=[128, 128])])

def load_args():
    args = p.parse_args()

    args.sim_perturbation = False
    if args.rand_perturbation or args.const_perturbation:
        args.sim_perturbation = True

    return args

if __name__ == "__main__":
    options = p.parse_args()

    print(options)
    print("----------")
    print(p.format_help())
    print("----------")
    print(p.format_values())