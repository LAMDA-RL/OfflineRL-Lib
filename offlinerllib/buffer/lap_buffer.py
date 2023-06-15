from typing import Optional, Union
from typing import Dict as DictLike
import numpy as np

from UtilsRL.data_structure import SumTree as CSumTree
from UtilsRL.data_structure import MinTree as CminTree
from UtilsRL.rl.buffer import TransitionSimpleReplay

def lap_propotional(metric_value, alpha, min_priority):
    return np.power(np.clip(metric_value, a_min=min_priority, a_max=None), alpha)


class LAPBuffer(TransitionSimpleReplay):
    def __init__(
        self, 
        max_size: int, 
        field_specs: Optional[DictLike]=None, 
        prioritized: bool=True, 
        alpha: float=0.4, 
        min_priority: float=1.0, 
        *args, **kwargs 
    ) -> None:
        super().__init__(max_size, field_specs, *args, **kwargs)
        self._prioritized = prioritized
        if prioritized:
            self.sum_tree = CSumTree(self._max_size)
            self.min_tree = CminTree(self._max_size)
            self.metric_fn = lambda x: lap_propotional(x, alpha, min_priority)
            self.max_metric_value = 1
        
        self.reset()
        
    def reset(self):
        super().reset()
        if self._prioritized:
            self.sum_tree.reset()
            self.min_tree.reset()
        
    def add_sample(self, data_dict: DictLike):
        data_len = None
        index_to_go = None
        for (_key, _data)in data_dict.items():
            if data_len is None:
                data_len = np.prod(_data.shape) // _data.shape[-1]
                index_to_go = np.arange(self._pointer, self._pointer+data_len) % self._max_size
            self.fields[_key][index_to_go] = _data
        self._pointer = (self._pointer + data_len) % self._max_size
        self._size = min(self._size + data_len, self._max_size)
        
        # update credential
        if self._prioritized:
            self.sum_tree.add(np.full(shape=[data_len, ], fill_value=self.max_metric_value))
            self.min_tree.add(np.full(shape=[data_len, ], fill_value=-self.max_metric_value))
        
    def random_batch(self, batch_size: int, return_idx: bool=True):
        if len(self) == 0:
            batch_data, batch_is, batch_idx = None, None, None
        else:
            if batch_size is None:
                raise NotImplementedError(f"you must specify a batch size for PER for now.")
            elif self._prioritized:
                batch_target = np.random.random(size=[batch_size, ])
                batch_idx, batch_p = self.sum_tree.find(batch_target)
                batch_idx = np.asarray(batch_idx)
            else:
                batch_idx = np.random.choice(self._size, size=batch_size)
            batch_data = {
                _key: self.fields[_key][batch_idx] for _key in self.fields
            }
        return (batch_data, batch_idx) if return_idx else batch_data
        
    def batch_update(self, batch_idx, metric_value):
        if self._prioritized:
            batch_idx = np.asarray(batch_idx)
            if len(batch_idx.shape) == 0:
                batch_idx = np.asarray([batch_idx, ])
            metric_value = np.asarray(metric_value)
            if len(metric_value.shape) == 0:
                metric_value = np.asarray([metric_value, ])
            # update crendential
            metric_value = self.metric_fn(metric_value)
            self.max_metric_value = max(np.max(metric_value), self.max_metric_value)
            self.sum_tree.update(batch_idx, metric_value)
            self.min_tree.update(batch_idx, -metric_value)
        
    def reset_max_priority(self):
        if self._prioritized:
            max_priority = - self.min_tree.min()
            self.max_metric_value = max_priority

