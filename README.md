# OfflineRL-Lib
<!-- > 🚧 This repo is not ready for release, benchmarking is ongoing. 🚧 -->
OfflineRL-Lib provides unofficial and benchmarked PyTorch implementations for selected OfflineRL algorithms, including: 
- [In-Sample Actor Critic (InAC)](https://arxiv.org/abs/2302.14372)
- [Extreme Q-Learning (XQL)](https://arxiv.org/abs/2301.02328)
- [Implicit Q-Learning (IQL)](https://arxiv.org/abs/2110.06169)
- [Decision Transformer (DT)](https://arxiv.org/abs/2106.01345)
- [Advantage-Weighted Actor Critic (AWAC)](https://arxiv.org/abs/2006.09359)
- [TD3-BC](https://arxiv.org/pdf/2106.06860.pdf)
- [TD7](https://arxiv.org/abs/2306.02451)

For Model-Based algorithms, please check [OfflineRL-Kit](https://github.com/yihaosun1124/OfflineRL-Kit)!


## Benchmark Results
<!-- See [reproduce/benchmark_result.md](https://github.com/typoverflow/OfflineRL-Lib/blob/master/reproduce/benchmark_result.md) for details.  -->

+ We benchmark and visualize the result via WandB. Click the following WandB links, and group the runs via the entry `task` (for offline experiments) or `env` (for online experiments). 
+ Available Runs
  + Offline: 
    + TD7 [:chart_with_upwards_trend:](https://wandb.ai/lamda-rl/TD7-D4RL)
    + XQL [:chart_with_upwards_trend:](https://wandb.ai/lamda-rl/XQL-D4RL)
    + InAC [:chart_with_upwards_trend:](https://wandb.ai/lamda-rl/InAC-D4RL)
    + AWAC [:chart_with_upwards_trend:](https://wandb.ai/lamda-rl/AWAC-D4RL)
    + IQL [:chart_with_upwards_trend:](https://wandb.ai/lamda-rl/IQL-D4RL)
    + TD3BC [:chart_with_upwards_trend:](https://wandb.ai/lamda-rl/TD3BC-Offline)
    + Decision Transformer [:chart_with_upwards_trend:](https://wandb.ai/lamda-rl/DecisionTransformer-Offline)
  + Online Runs
    + SAC [:chart_with_upwards_trend:](https://wandb.ai/lamda-rl/SAC-Online)
    + TD3 [:chart_with_upwards_trend:](https://wandb.ai/lamda-rl/TD3-Online)
    + TD7 [:chart_with_upwards_trend:](https://wandb.ai/lamda-rl/TD7-Online)
    + XSAC [:chart_with_upwards_trend:](https://wandb.ai/lamda-rl/XSAC-Online)

## Citing OfflineRL-Lib
If you use OfflineRL-Lib in your work, please use the following bibtex
```tex
@misc{offinerllib,
  author = {Chenxiao Gao},
  title = {OfflineRL-Lib: Benchmarked Implementations of Offline RL Algorithms},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/typoverflow/OfflineRL-Lib}},
}
```

## Acknowledgements
We thank [CORL](https://github.com/tinkoff-ai/CORL) for providing finetuned hyper-parameters. 