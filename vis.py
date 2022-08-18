import numpy as np
import torch.utils.data
from configs.config_loader import load_args
import ipdb
from learning_algs.FC_Net import Net
from environments.humanoid.humanoid_treadmill_env import HumanoidTreadmillEnv
import pandas as pd

if __name__ == '__main__':
    args = load_args()

    env = HumanoidTreadmillEnv(args=args) #, n_envs=1)

    # model = Net(env.observation_space.shape[0], env.action_space.shape[0], args.hidden_dim)
    # model.load_state_dict(torch.load('results/models/no_phase/no_phase_iter5100.pt'))  # relative file path
    # model.cuda()

    while True:
        env.reset()
        done = False

        while not done:
            # print(env.muscle_torque_offset)

            with torch.no_grad():
                #act = model.sample_best_actions(torch.from_numpy(state).cuda().type(torch.cuda.FloatTensor)).cpu().numpy()
                act = np.zeros(env.action_space.shape[0])

            next_state, reward, done, info = env.step(act)
            env.render()
            # time.sleep(2)
            state = next_state
