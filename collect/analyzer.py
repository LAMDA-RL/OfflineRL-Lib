import numpy as np
import matplotlib.pyplot as plt

path = 'datasets/variant-world/gravity-12/Walker2d-v3/seed0/'

with np.load(path + 'data.npz') as data:
    plt.hist(data['episode_return'], bins=100)
    # title
    plt.title('gravity-12/Walker2d-v3')
    # save to path/episode_return.png
    plt.savefig(path + 'episode_return.png')
