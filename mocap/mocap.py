import numpy as np
from pathlib import Path
import pprint

from typing import List
from typing import Dict
from typing import Tuple
from scipy.interpolate import interp1d

class MoCap:
    def __init__(self, path, args=None) -> None:

        self.percent_train = args.percent_train
        self.train_data, self.val_data = self.load_data(path)
        # print(self.train_data.shape)
        self.obs_size = self.train_data.shape[1] - args.num_frc
        self.frc_size = args.num_frc

    def load_data(self, path) -> Tuple[np.ndarray, np.ndarray]:

        # Stack all processed data
        data = np.array([])
        for np_name in Path(path).glob('*.np[yz]'):
            traj = np.load(np_name)
            data = np.vstack([data, traj]) if data.size else traj

        assert len(data) > 0, 'No data loaded'

        # Split into train and validation sets
        n = int(len(data) * self.percent_train)

        train = data[:n, :]
        val = data[n:, :]

        return train, val

    def shuffle_split(self, approx_batch_size):
        d_train = max(1, len(self.train_data) // approx_batch_size)
        d_val = max(1, len(self.val_data) // approx_batch_size)

        np.random.shuffle(self.train_data)

        trn = np.array_split(self.train_data, d_train, axis=0)
        val = np.array_split(self.val_data, d_val, axis=0)

        return trn, val

if __name__ == "__main__":
    path = '/home/takaraet/fc_modeling/mocap/mocap_data/humanoid/Subject2'
    mocap = MoCap(path)
    trn, val = mocap.shuffle_split(50)
