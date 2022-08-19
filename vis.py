import numpy as np
import torch.utils.data
from configs.config_loader import load_args
import ipdb
from learning_algs.FC_Net import Net
from environments.humanoid.humanoid_treadmill_env import HumanoidTreadmillEnv
import pandas as pd
from utils.load_envs import load_environments
from learning_algs.amp_models import ActorCriticNet

if __name__ == '__main__':
    args = load_args()

    # _, env, _ = load_environments(args)
    env = HumanoidTreadmillEnv(args=args)
    model = ActorCriticNet(env.observation_space.shape[0], env.action_space.shape[0], [128, 128])


    model.load_state_dict(torch.load('results/models/fc_test/fc_test_iter3400.pt'))  # relative file path
    model.cuda()
    # print(env.action_space.shape[0])

    while True:
        state = env.reset()
        done = False

        while not done:
            with torch.no_grad():
                act = model.sample_best_actions(torch.from_numpy(state).cuda().type(torch.cuda.FloatTensor)).cpu().numpy()
                # act = np.zeros(env.action_space.shape[0])

            next_state, reward, done, info = env.step(act)
            # print("in loop")
            env.render()
            # time.sleep(2)
            state = next_state
