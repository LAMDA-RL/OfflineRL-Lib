import numpy as np
import matplotlib.pyplot as plt

path = 'datasets/variant-world/sac-mass-5/Lift/seed0/'

with np.load(path + 'data.npz') as data:
    plt.hist(data['episode_return'], bins=100)
    # title
    plt.title('Lift-mass-5')
    # save to path/episode_return.png
    plt.savefig(path + 'episode_return.png')
