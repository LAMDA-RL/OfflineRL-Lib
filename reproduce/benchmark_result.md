# Benchmark Results
When certain design choices, e.g. the choice of autodiff backend (jax or tf or pytorch) vary, the preference for each hyper-parameters may vary as well. Hence when benchmarking, we tested each algorithm's performace in three ways: 
+ **Paper Performance**: the performance reported in white paper;
+ **OfflineRL-Lib (with paper args)**: the performance obtained by using OfflineRL-Lib implementation and the configs in paper or original implementations;
+ **OfflineRL-Lib (with CORL args)**: the performance obtained by using OfflineRL-Lib implementation and the configs in [CORL](https://github.com/tinkoff-ai/CORL). 

> For the last option, arguments are directly borrowed from CORL. CORL provides simplified single-file implementations of these algorithms as well as their finetuned hyper-parameters based on pytorch, please check [their repo](https://github.com/tinkoff-ai/CORL) as well. 

## InAC [:page_facing_up:](https://arxiv.org/abs/2302.14372) [:chart_with_upwards_trend:](https://wandb.ai/lamda-rl/InAC-Offline)
Note that:
+ For each run, we record the averaged (over 5 independent runs) best score for the last two checkpoints, and report the higher number between them. We think limited model selection (only two deployments) is permitted; however if you need to evaluate the models otherwisely, please refer to the wandb logs. 
<table>
    <thead>
        <tr>
            <th>Task</th>
            <th>Dataset Quality</th>
            <th>Paper Performance</th>
            <th>OfflineRL-Lib<br>(paper args)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=4>halfcheetah</td>
            <td>medium-v2</td><td>48.30±0.02</td><td>47.31±1.03</td>
        </tr>
        <tr>
            <td>medium-replay-v2</td><td>44.30±0.02</td><td>43.73±1.19</td>
        </tr>
        <tr>
            <td>medium-expert-v2</td><td>83.50±0.34</td><td>92.04±1.14</td>
        </tr>
        <tr>
            <td>expert-v2</td><td>93.60±0.04</td><td>94.26±1.41</td>
        </tr>
        <tr>
            <td rowspan=4>hopper</td>
            <td>medium-v2</td><td>60.3±0.20</td><td>76.97±11.32</td>
        </tr>
        <tr>
            <td>medium-replay-v2</td><td>92.10±0.38</td><td>84.99±6.19</td>
        </tr>
        <tr>
            <td>medium-expert-v2</td><td>93.80±0.69</td><td>101.96±7.63</td>
        </tr>
        <tr>
            <td>expert-v2</td><td>103.40±0.38</td><td>100.42±20.19</td>
        </tr>
        <tr>
            <td rowspan=4>walker2d</td>
            <td>medium-v2</td><td>71.1±0.53</td><td>79.65±9.24</td>
        </tr>
        <tr>
            <td>medium-replay-v2</td><td>69.80±0.57</td><td>77.55±2.26</td>
        </tr>
        <tr>
            <td>medium-expert-v2</td><td>109.00±0.10</td><td>110.84±0.48</td>
        </tr>
        <tr>
            <td>expert-v2</td><td>110.60±0.09</td><td>111.38±1.27</td>
        </tr>
    </tbody>
</table>



## XQL [:page_facing_up:](https://arxiv.org/abs/2301.02328) [:chart_with_upwards_trend:](https://wandb.ai/lamda-rl/XQL-Offline)
Note that:
+ For each run, we record the averaged (over 5 independent runs) best score for the last two checkpoints, and report the higher number between them. We think limited model selection (only two deployments) is permitted; however if you need to evaluate the models otherwisely, please refer to the wandb logs. 

<table>
    <thead>
        <tr>
            <th>Task</th>
            <th>Dataset Quality</th>
            <th>Paper Performance<br>(consistent)</th>
            <th>Paper Performance<br>(tuned)</th>
            <th>OfflineRL-Lib<br>(paper args)<br>(consistent)</th>
            <th>OfflineRL-Lib<br>(paper args)<br>(tuned)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=4>halfcheetah</td>
            <td>random-v2</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td>
        </tr>
        <tr>
            <td>medium-v2</td><td>47.7</td><td>48.3</td><td>NA</td><td>47.9±0.3</td>
        </tr>
        <tr>
            <td>medium-replay-v2</td><td>44.8</td><td>45.2</td><td>NA</td><td>44.3±0.4</td>
        </tr>
        <tr>
            <td>medium-expert-v2</td><td>89.8</td><td>94.2</td><td>NA</td><td>92.1±1.0</td>
        </tr>
        <tr>
            <td rowspan=4>hopper</td>
            <td>random-v2</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td>
        </tr>
        <tr>
            <td>medium-v2</td><td>71.1</td><td>74.2</td><td>NA</td><td>67.0±6.8</td>
        </tr>
        <tr>
            <td>medium-replay-v2</td><td>97.3</td><td>100.7</td><td>NA</td><td>96.9±6.2</td>
        </tr>
        <tr>
            <td>medium-expert-v2</td><td>107.1</td><td>111.2</td><td>NA</td><td>101.9±5.2</td>
        </tr>
        <tr>
            <td rowspan=4>walker2d</td>
            <td>random-v2</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td>
        </tr>
        <tr>
            <td>medium-v2</td><td>81.5</td><td>84.2</td><td>NA</td><td>83.8±0.4</td>
        </tr>
        <tr>
            <td>medium-replay-v2</td><td>75.9</td><td>82.2</td><td>NA</td><td>76.5±5.2</td>
        </tr>
        <tr>
            <td>medium-expert-v2</td><td>110.1</td><td>112.7</td><td>NA</td><td>110.7±0.6</td>
        </tr>
    </tbody>
</table>


## IQL [:page_facing_up:](https://arxiv.org/abs/2110.06169) [:chart_with_upwards_trend:](https://wandb.ai/lamda-rl/IQL-Offline)

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
            <td rowspan=6>halfcheetah</td>
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
            <td>full-replay-v2</td><td>NA</td><td>73.5±0.8</td><td>74.9±0.3</td>
        </tr>
        <tr>
            <td>expert-v2</td><td>NA</td><td>94.8±0.4</td><td>95.7±2.6</td>
        </tr>
        <tr>
            <td rowspan=6>hopper</td>
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
            <td>full-replay-v2</td><td>NA</td><td>104.5±6.0</td><td>106.3±1.0</td>
        </tr>
        <tr>
            <td>expert-v2</td><td>NA</td><td>110.1±0.8</td><td>103.8±7.9</td>
        </tr>
        <tr>
            <td rowspan=6>walker2d</td>
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
        <tr>
            <td>full-replay-v2</td><td>NA</td><td>92.9±3.5</td><td>99.2±0.7</td>
        </tr>
        <tr>
            <td>expert-v2</td><td>NA</td><td>109.7±0.3</td><td>112.6±0.4</td>
        </tr>
    </tbody>
</table>

## Decision Transformer [:page_facing_up:](https://arxiv.org/abs/2106.01345) [:chart_with_upwards_trend:](https://wandb.ai/lamda-rl/DecisionTransformer-Offline)
Note that
+ For each task, we tested with two init return-to-go: [6000, 12000] for halfcheetah, [1800, 3600] for hopper and [2500, 5000] for walker2d.
+ For each run, we record the averaged (over 5 independent runs) best score for the last two checkpoints, and report the higher number between them. We think limited model selection (only two deployments) is permitted; however if you need to evaluate the models otherwisely, please refer to the wandb logs. 

<table>
    <thead>
        <tr>
            <th>Task</th>
            <th>Dataset Quality</th>
            <th>Paper Performance</th>
            <th>OfflineRL-Lib<br>(with CORL args)
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=3>halfcheetah</td>
            <td>medium-v2</td><td>42.6±0.1</td><td>42.1±0.4 (6000), 41.9±0.6 (12000)</td>
        </tr>
        <tr>
            <td>medium-replay-v2</td><td>36.6±0.8</td><td>38.8±2.2 (6000), 32.2±3.9 (12000)</td>
        </tr>
        <tr>
            <td>medium-expert-v2</td><td>86.8±1.3</td><td>43.2±2.2 (6000), 92.1±1.1 (12000)</td>
        </tr>
        <tr>
            <td rowspan=3>hopper</td>
            <td>medium-v2</td><td>67.6±1.0</td><td>56.4±1.6 (1800), 59.1±2.7 (3600)</td>
        </tr>
        <tr>
            <td>medium-replay-v2</td><td>82.7±7.0</td><td>56.0±10.4 (1800), 82.5±8.4 (3600)</td>
        </tr>
        <tr>
            <td>medium-expert-v2</td><td>107.6±1.8</td><td>55.3±1.2 (1800), 109.3±2.9 (3600)</td>
        </tr>
        <tr>
            <td rowspan=3>walker2d</td>
            <td>medium-v2</td><td>74.0±1.4</td><td>68.8±8.1 (2500), 74.5±2.9 (5000)</td>
        </tr>
        <tr>
            <td>medium-replay-v2</td><td>66.6±3.0</td><td>52.6±5.6 (2500), 58.8±9.4 (5000)</td>
        </tr>
        <tr>
            <td>medium-expert-v2</td><td>108.1±0.2</td><td>66.6±3.4 (2500), 107.4±3.1 (5000)</td>
        </tr>
    </tbody>
</table>

## AWAC [:page_facing_up:](https://arxiv.org/abs/2006.09359) [:chart_with_upwards_trend:](https://wandb.ai/lamda-rl/AWAC-Offline)
Note that
+ For each run, we record the averaged (over 5 independent runs) best score for the last two checkpoints, and report the higher number between them. We think limited model selection (only two deployments) is permitted; however if you need to evaluate the models otherwisely, please refer to the wandb logs. 

<table>
    <thead>
        <tr>
            <th>Task</th>
            <th>Dataset Quality</th>
            <th>Paper Performance<br>(maybe on -v0)</th>
            <th>OfflineRL-Lib<br>(with CORL args)
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=6>halfcheetah</td>
            <td>random-v2</td><td>2.2</td><td>8.2±3.8</td>
        </tr>
        <tr>
            <td>medium-v2</td><td>37.4</td><td>49.8±0.4</td>
        </tr>
        <tr>
            <td>medium-replay-v2</td><td>NA</td><td>46.5±0.4</td>
        </tr>
        <tr>
            <td>medium-expert-v2</td><td>36.8</td><td>93.5±2.7</td>
        </tr>
        <tr>
            <td>full-replay-v2</td><td>NA</td><td>76.9±1.3</td>
        </tr>
        <tr>
            <td>expert-v2</td><td>78.5</td><td>99.9±0.5</td>
        </tr>
        <tr>
            <td rowspan=6>hopper</td>
            <td>random-v2</td><td>9.6</td><td>18.3±12.2</td>
        </tr>
        <tr>
            <td>medium-v2</td><td>72.0</td><td>64.8±6.0</td>
        </tr>
        <tr>
            <td>medium-replay-v2</td><td>NA</td><td>91.7±16.0</td>
        </tr>
        <tr>
            <td>medium-expert-v2</td><td>80.9</td><td>86.4±25.7</td>
        </tr>
        <tr>
            <td>full-replay-v2</td><td>NA</td><td>107.9±6.3</td>
        </tr>
        <tr>
            <td>expert-v2</td><td>85.2</td><td>94.7±8.0</td>
        </tr>
        <tr>
            <td rowspan=6>walker2d</td>
            <td>random-v2</td><td>5.1</td><td>6.6±6.0</td>
        </tr>
        <tr>
            <td>medium-v2</td><td>30.1</td><td>86.5±1.0</td>
        </tr>
        <tr>
            <td>medium-replay-v2</td><td>NA</td><td>87.6±4.3</td>
        </tr>
        <tr>
            <td>medium-expert-v2</td><td>42.7</td><td>113.3±0.5</td>
        </tr>
        <tr>
            <td>full-replay-v2</td><td>NA</td><td>99.6±1.5</td>
        </tr>
        <tr>
            <td>expert-v2</td><td>57</td><td>113.0±0.4</td>
        </tr>
    </tbody>
</table>

## TD3BC [:page_facing_up:](https://arxiv.org/abs/2106.06860) [:chart_with_upwards_trend:](https://wandb.ai/lamda-rl/TD3BC-Offline)
Note that
+ For each run, we record the averaged (over 5 independent runs) best score for the last two checkpoints, and report the higher number between them. We think limited model selection (only two deployments) is permitted; however if you need to evaluate the models otherwisely, please refer to the wandb logs. 

<table>
    <thead>
        <tr>
            <th>Task</th>
            <th>Dataset Quality</th>
            <th>Paper Performance<br></th>
            <th>OfflineRL-Lib<br>(with CORL args)
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=4>halfcheetah</td>
            <td>random-v2</td><td>10.2±1.3</td><td>11.5±1.0</td>
        </tr>
        <tr>
            <td>medium-v2</td><td>41.3±0.5</td><td>48.5±0.4</td>
        </tr>
        <tr>
            <td>medium-replay-v2</td><td>43.3±0.9</td><td>44.7±0.7</td>
        </tr>
        <tr>
            <td>medium-expert-v2</td><td>97.9±4.4</td><td>91.7±3.2</td>
        </tr>
        <tr>
            <td rowspan=4>hopper</td>
            <td>random-v2</td><td>11.0±0.2</td><td>10.3±2.8</td>
        </tr>
        <tr>
            <td>medium-v2</td><td>99.5±1.0</td><td>61.8±1.6</td>
        </tr>
        <tr>
            <td>medium-replay-v2</td><td>31.4±3.0</td><td>73.5±20.4</td>
        </tr>
        <tr>
            <td>medium-expert-v2</td><td>112.2±0.2</td><td>103.4±8.9</td>
        </tr>
        <tr>
            <td rowspan=4>walker2d</td>
            <td>random-v2</td><td>1.4±1.6</td><td>1.95±2.6</td>
        </tr>
        <tr>
            <td>medium-v2</td><td>79.7±1.8</td><td>84.3±1.4</td>
        </tr>
        <tr>
            <td>medium-replay-v2</td><td>25.2±5.1</td><td>84.8±4.2</td>
        </tr>
        <tr>
            <td>medium-expert-v2</td><td>101.1±9.3</td><td>110.4±0.57</td>
        </tr>
    </tbody>
</table>