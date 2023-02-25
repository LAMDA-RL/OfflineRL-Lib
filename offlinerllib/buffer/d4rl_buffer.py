import numpy as np
from torch.utils.data import Dataset, IterableDataset

from offlinerllib.buffer.base import Buffer
from offlinerllib.utils.functional import discounted_cum_sum


def pad_along_axis(
    arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0
) -> np.ndarray:
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)
    return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)


class D4RLTransitionBuffer(Buffer, IterableDataset, Dataset):
    def __init__(self, dataset):
        self.observations = dataset["observations"].astype(np.float32)
        self.actions = dataset["actions"].astype(np.float32)
        self.rewards = dataset["rewards"][:, None].astype(np.float32)
        self.terminals = dataset["terminals"][:, None].astype(np.float32)
        self.next_observations = dataset["next_observations"].astype(np.float32)
        self.size = len(dataset["observations"])
        self.masks = np.ones([self.size, 1], dtype=np.float32)
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            "observations": self.observations[idx], 
            "actions": self.actions[idx], 
            "rewards": self.rewards[idx], 
            "terminals": self.terminals[idx], 
            "next_observations": self.next_observations[idx], 
            "masks": self.masks[idx]
        }
        
    def __iter__(self):
        while True:
            idx = np.random.randint(self.size)
            yield self.__getitem__(idx)
        
    def random_batch(self, batch_size: int):
        idx = np.random.randint(self.size, size=batch_size)
        return self.__getitem__(idx)
        

class D4RLTrajectoryBuffer(Buffer, IterableDataset):
    def __init__(self, dataset, seq_len: int, discount: float=1.0, return_scale: float=1.0):
        # fetch data from dataset
        converted_dataset = {
            "observations": dataset["observations"].astype(np.float32), 
            "actions": dataset["actions"].astype(np.float32), 
            "rewards": dataset["rewards"][:, None].astype(np.float32), 
            "terminals": dataset["terminals"][:, None].astype(np.float32), 
            "next_observations": dataset["next_observations"].astype(np.float32)
        }

        traj, traj_len = [], []
        self.seq_len = seq_len
        traj_start = 0
        for i in range(dataset["rewards"].shape[0]):
            if dataset["ends"][i]:
                assert dataset["terminals"][i] or i+1-traj_start == 999 or i==dataset["rewards"].shape[0]-1
                episode_data = {k: v[traj_start:i+1] for k, v in converted_dataset.items()}
                episode_data["returns"] = discounted_cum_sum(episode_data["rewards"], discount=discount) / return_scale
                traj.append(episode_data)
                traj_len.append(i+1-traj_start)
                traj_start = i+1
        self.traj = np.array(traj, dtype=object)
        self.traj_len = np.array(traj_len)
        self.traj_num = len(self.traj_len)
        self.size = self.traj_len.sum()
        self.sample_prob = self.traj_len / self.size

        del converted_dataset
        
    def __prepare_sample(self, traj_idx, start_idx):
        traj = self.traj[traj_idx]
        sample = {k: v[start_idx:start_idx+self.seq_len] for k, v in traj.items()}
        sample_len = len(sample["observations"])
        if sample_len < self.seq_len:
            sample = {k: pad_along_axis(v, pad_to=self.seq_len) for k, v in sample.items()}
        masks = np.hstack([np.ones(sample_len), np.zeros(self.seq_len-sample_len)])
        sample["masks"] = masks
        sample["timesteps"] = np.arange(start_idx, start_idx+self.seq_len)
        return sample
    
    def __iter__(self):
        while True:
            traj_idx = np.random.choice(self.traj_num, p=self.sample_prob)
            start_idx = np.random.choice(self.traj_len[traj_idx])
            yield self.__prepare_sample(traj_idx, start_idx)
        
    def random_batch(self, batch_size: int):
        batch_data = {}
        traj_idxs = np.random.choice(self.traj_num, size=batch_size, p=self.sample_prob)
        for i in range(batch_size):
            traj_idx = traj_idxs[i]
            start_idx = np.random.choice(self.traj_len[traj_idx])
            sample = self.__prepare_sample(traj_idx, start_idx)
            for _key, _value in sample.items():
                if not _key in batch_data:
                    batch_data[_key] = []
                batch_data[_key].append(_value)
        for _key, _value in batch_data.items():
            batch_data[_key] = np.vstack(_value)
        return batch_data
            
