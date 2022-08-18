import wandb
from configs.config_loader import load_args
from learning_algs.FC_Net import FC_Net

if __name__ == '__main__':
    args = load_args()

    model = FC_Net(args)
    print(model)
    model.train()





