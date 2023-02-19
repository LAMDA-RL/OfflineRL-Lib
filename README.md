# OfflineRL-Lib
> 🚧 This repo is not ready for release, benchmarking is ongoing. 🚧

OfflineRL-Lib provides unofficial and benchmarked PyTorch implementations for selected OfflineRL algorithms, including: 
- [Implicit Q-Learning](https://arxiv.org/abs/2110.06169)
- [Extreme Q-Learning](https://arxiv.org/abs/2301.02328)

Currently OfflineRL-Lib is heavily based off [UtilsRL](https://github.com/typoverflow/UtilsRL), and we will release a standalone version once it is ready. 


## Benchmark Results
When certain design choices, e.g. the choice of autodiff backend (jax or tf or pytorch) vary, the preference for each hyper-parameters may vary as well. Hence when benchmarking, we tested each algorithm's performace in three ways: 
+ **Paper Performance**: the performance reported in white paper;
+ **OfflineRL-Lib (with paper args)**: the performance obtained by using OfflineRL-Lib implementation and the configs in paper or original implementations;
+ **OfflineRL-Lib (with CORL args)**: the performance obtained by using OfflineRL-Lib implementation and the configs in [CORL](https://github.com/tinkoff-ai/CORL). 

> For the last option, arguments are directly borrowed from CORL. CORL provides simplified single-file implementations of these algorithms as well as their finetuned hyper-parameters based on pytorch, please check [their repo](https://github.com/tinkoff-ai/COR) as well. 

### IQL [:page_facing_up:](https://arxiv.org/abs/2110.06169) [:chart_with_upwards_trend:](https://wandb.ai/lamda-rl/IQL-Offline)

<table>
    <thead>
        <tr>
            <th>Task</th>
            <th>Dataset Quality</th>
            <th>Paper Performance</th>
            <th>OfflineRL-Lib<br>(with paper args)</th>
            <th>OfflineRL-Lib<br>(with CORL args)
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=4>halfcheetah</td>
            <td>random-v2</td><td>NA</td><td>9.4±3.9</td><td>13.5±3.9</td>
        </tr>
        <tr>
            <td>medium-v2</td><td>47.4</td><td>47.3±0.2</td><td>48.6±0.2</td>
        </tr>
        <tr>
            <td>medium-replay-v2</td><td>44.2</td><td>43.7±0.7</td><td>44.3±0.4</td>
        </tr>
        <tr>
            <td>medium-expert-v2</td><td>86.7</td><td>89.7±2.9</td><td>93.9±1.6</td>
        </tr>
        <tr>
            <td rowspan=4>hopper</td>
            <td>random-v2</td><td>NA</td><td>7.9±0.3</td><td>7.3±0.1</td>
        </tr>
        <tr>
            <td>medium-v2</td><td>66.3</td><td>64.8±7.2</td><td>54.5±1.6</td>
        </tr>
        <tr>
            <td>medium-replay-v2</td><td>94.7</td><td>93.4±7.9</td><td>41.5±23.0</td>
        </tr>
        <tr>
            <td>medium-expert-v2</td><td>91.5</td><td>97.8±9.0</td><td>108.1±3.1</td>
        </tr>
        <tr>
            <td rowspan=4>walker2d</td>
            <td>random-v2</td><td>NA</td><td>6.0±1.0</td><td>3.1±0.9</td>
        </tr>
        <tr>
            <td>medium-v2</td><td>78.3</td><td>83.5±2.2</td><td>81.3±8.7</td>
        </tr>
        <tr>
            <td>medium-replay-v2</td><td>73.9</td><td>66.6±16.2</td><td>77.0±7.3</td>
        </tr>
        <tr>
            <td>medium-expert-v2</td><td>109.6</td><td>108.9±2.5</td><td>112.4±0.8</td>
        </tr>
    </tbody>
</table>