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


class D4RLTransitionBuffer(Buffer, IterableDataset):
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
    def __init__(
        self, 
        dataset, 
        seq_len: int, 
        discount: float=1.0, 
        return_scale: float=1.0,
    ) -> None:
        converted_dataset = {
            "observations": dataset["observations"].astype(np.float32), 
            "actions": dataset["actions"].astype(np.float32), 
            "rewards": dataset["rewards"][:, None].astype(np.float32), 
            "terminals": dataset["terminals"][:, None].astype(np.float32), 
            "next_observations": dataset["next_observations"].astype(np.float32)
        }
        traj, traj_len = [], []
        self.seq_len = seq_len
        self.discount = discount
        self.return_scale = return_scale
        traj_start = 0
        for i in range(dataset["rewards"].shape[0]):
            if dataset["ends"][i]:
                episode_data = {k: v[traj_start:i+1] for k, v in converted_dataset.items()}
                episode_data["returns"] = discounted_cum_sum(episode_data["rewards"], discount=discount) * self.return_scale
                traj.append(episode_data)
                traj_len.append(i+1-traj_start)
                traj_start = i+1
        self.traj_len = np.array(traj_len)
        self.size = self.traj_len.sum()
        self.traj_num = len(self.traj_len)
        self.sample_prob = self.traj_len / self.size
        
        # pad trajs to have the same mask len
        self.max_len = self.traj_len.max() + self.seq_len - 1  # this is for the convenience of sampling
        for i_traj in range(self.traj_num):
            this_len = self.traj_len[i_traj]
            for _key, _value in traj[i_traj].items():
                traj[i_traj][_key] = pad_along_axis(_value, pad_to=self.max_len)
            traj[i_traj]["masks"] = np.hstack([np.ones(this_len), np.zeros(self.max_len-this_len)])
        
        # register all entries
        self.observations = np.asarray([t["observations"] for t in traj])
        self.actions = np.asarray([t["actions"] for t in traj])
        self.rewards = np.asarray([t["rewards"] for t in traj])
        self.terminals = np.asarray([t["terminals"] for t in traj])
        self.next_observations = np.asarray([t["next_observations"] for t in traj])
        self.returns = np.asarray([t["returns"] for t in traj])
        self.masks = np.asarray([t["masks"] for t in traj])
        self.timesteps = np.arange(self.max_len)

    def __len__(self):
        return self.size

    def __prepare_sample(self, traj_idx, start_idx):
        return {
            "observations": self.observations[traj_idx, start_idx:start_idx+self.seq_len], 
            "actions": self.actions[traj_idx, start_idx:start_idx+self.seq_len], 
            "rewards": self.rewards[traj_idx, start_idx:start_idx+self.seq_len], 
            "terminals": self.terminals[traj_idx, start_idx:start_idx+self.seq_len], 
            "next_observations": self.next_observations[traj_idx, start_idx:start_idx+self.seq_len], 
            "returns": self.returns[traj_idx, start_idx:start_idx+self.seq_len], 
            "masks": self.masks[traj_idx, start_idx:start_idx+self.seq_len], 
            "timesteps": self.timesteps[start_idx:start_idx+self.seq_len]
        }
    
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