from abc import ABC, abstractmethod
from typing import Dict, Any

class Replay(ABC):
    @abstractmethod
    def random_batch(self, batch_size: int) -> Dict[str, Any]:
        raise NotImplementedError