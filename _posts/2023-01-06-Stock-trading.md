---
layout: post
title: Stock-trading
author: [MOCI2001]
category: [Final-Project]
tags: [jekyll, ai]
---

Stock trading proposes an automatic trading system that combines neural networks and reinforcement learning to determine trading signals and trading position sizes. Use stock market data, including daily opening price, closing price, highest price, lowest price, trading volume. The profitability of the system is positive under the long-term test, which means that the system has a good performance in predicting the rise and fall of stocks.

---
## Introduction of Reinforcement Learning
**Blog:** [Key Concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)<br>
**Blog:** [A (Long) Peek into Reinforcement Learning](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html)<br>

<img src="https://spinningup.openai.com/en/latest/_images/rl_diagram_transparent_bg.png">
<img width="50%" height="50%" src="https://www.tensorflow.org/agents/tutorials/images/rl_overview.png">

---
### What is Reinforcement Learning ?
**Blog:** [李宏毅老師 Deep Reinforcement Learning (2017 Spring)【筆記】](https://medium.com/change-the-world-with-technology/%E6%9D%8E%E5%AE%8F%E6%AF%85%E8%80%81%E5%B8%AB-deep-reinforcement-learning-2017-spring-%E7%AD%86%E8%A8%98-3784ddb23e0)<br>

<iframe width="560" height="315" src="https://www.youtube.com/embed/XWukX-ayIrs" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<table>
<tr>
<td><img src="https://miro.medium.com/max/2000/1*JuRvWsTyaRWZaYVirSsWiA.png"></td>
<td><img src="https://miro.medium.com/max/2000/1*GMGAfQeLvxJnTRQOEuTMDw.png"></td>
</tr>
</table>

---
## Algorithms
### Taxonomy of RL Algorithms
**Blog:** [Kinds of RL Alogrithms](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)<br>

![](https://spinningup.openai.com/en/latest/_images/rl_algorithms_9_15.svg)

* **Value-based methods** : Deep Q Learning
  - Where we learn a value function that will map each state action pair to a value.
* **Policy-based methods** : Reinforce with Policy Gradients
  - where we directly optimize the policy without using a value function
  - This is useful when the action space is continuous (連續) or stochastic (隨機)
  - use total rewards of the episode 
* **Hybrid methods** : Actor-Critic
  - a Critic that measures how good the action taken is (value-based)
  - an Actor that controls how our agent behaves (policy-based)
* **Model-based methods** : Partially-Observable Markov Decision Process (POMDP)
  - State-transition models
  - Observation-transition models

---
### List of RL Algorithms
1. **Q-Learning** 
  - [An Analysis of Temporal-Difference Learning with Function Approximation](http://web.mit.edu/jnt/www/Papers/J063-97-bvr-td.pdf)
  - [Algorithms for Reinforcement Learning](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf)
  - [A Finite Time Analysis of Temporal Difference Learning With Linear Function Approximation](https://arxiv.org/abs/1806.02450)
2. **A2C** (Actor-Critic Algorithms): [Actor-Critic Algorithms](https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)
3. **DQN** (Deep Q-Networks): [1312.5602](https://arxiv.org/abs/1312.5602)
4. **TRPO** (Trust Region Policy Optimizaton): [1502.05477](https://arxiv.org/abs/1502.05477)
5. **DDPG** (Deep Deterministic Policy Gradient): [1509.02971](https://arxiv.org/abs/1509.02971)
6. **DDQN** (Deep Reinforcement Learning with Double Q-learning): [1509.06461](https://arxiv.org/abs/1509.06461)
7. **DD-Qnet** (Double Dueling Q Net): [1511.06581](https://arxiv.org/abs/1511.06581)
8. **A3C** (Asynchronous Advantage Actor-Critic): [1602.01783](https://arxiv.org/abs/1602.01783)
9. **ICM** (Intrinsic Curiosity Module): [1705.05363](https://arxiv.org/abs/1705.05363)
10. **I2A** (Imagination-Augmented Agents): [1707.06203](https://arxiv.org/abs/1707.06203)
11. **PPO** (Proximal Policy Optimization): [1707.06347](https://arxiv.org/abs/1707.06347)
12. **C51** (Categorical 51-Atom DQN): [1707.06887](https://arxiv.org/abs/1707.06887)
13. **HER** (Hindsight Experience Replay): [1707.01495](https://arxiv.org/abs/1707.01495)
14. **MBMF** (Model-Based RL with Model-Free Fine-Tuning): [1708.02596](https://arxiv.org/abs/1708.02596)
15. **Rainbow** (Combining Improvements in Deep Reinforcement Learning): [1710.02298](https://arxiv.org/abs/1710.02298)
16. **QR-DQN** (Quantile Regression DQN): [1710.10044](https://arxiv.org/abs/1710.10044)
17. **AlphaZero** : [1712.01815](https://arxiv.org/abs/1712.01815)
18. **SAC** (Soft Actor-Critic): [1801.01290](https://arxiv.org/abs/1801.01290)
19. **TD3** (Twin Delayed DDPG): [1802.09477](https://arxiv.org/abs/1802.09477)
20. **MBVE** (Model-Based Value Expansion): [1803.00101](https://arxiv.org/abs/1803.00101)
21. **World Models**: [1803.10122](https://arxiv.org/abs/1803.10122)
22. **IQN** (Implicit Quantile Networks for Distributional Reinforcement Learning): [1806.06923](https://arxiv.org/abs/1806.06923)
23. **SHER** (Soft Hindsight Experience Replay): [2002.02089](https://arxiv.org/abs/2002.02089)
24. **LAC** (Actor-Critic with Stability Guarantee): [2004.14288](https://arxiv.org/abs/2004.14288)
25. **AGAC** (Adversarially Guided Actor-Critic): [2102.04376](https://arxiv.org/abs/2102.04376)
26. **TATD3** (Twin actor twin delayed deep deterministic policy gradient learning for batch process control): [2102.13012](https://arxiv.org/abs/2102.13012)
27. **SACHER** (Soft Actor-Critic with Hindsight Experience Replay Approach): [2106.01016](https://arxiv.org/abs/2106.01016)
28. **MHER** (Model-based Hindsight Experience Replay): [2107.00306](https://arxiv.org/abs/2107.00306)

---
## Open Environments
**[Best Benchmarks for Reinforcement Learning: The Ultimate List](https://neptune.ai/blog/best-benchmarks-for-reinforcement-learning)**<br>
* [AI Habitat](https://aihabitat.org/) – Virtual embodiment; Photorealistic & efficient 3D simulator;
* [Behaviour Suite](https://github.com/deepmind/bsuite) – Test core RL capabilities; Fundamental research; Evaluate generalization;
* [DeepMind Control Suite](https://github.com/deepmind/dm_control) – Continuous control; Physics-based simulation; Creating environments;
* [DeepMind Lab](https://github.com/deepmind/lab) – 3D navigation; Puzzle-solving;
* [DeepMind Memory Task Suite](https://github.com/deepmind/dm_memorytasks) – Require memory; Evaluate generalization;
* [DeepMind Psychlab](https://github.com/deepmind/lab/tree/master/game_scripts/levels/contributed/psychlab) – Require memory; Evaluate generalization;
* [Google Research Football](https://github.com/google-research/football) – Multi-task; Single-/Multi-agent; Creating environments;
* [Meta-World](https://github.com/rlworkgroup/metaworld) – Meta-RL; Multi-task;
* [MineRL](https://minerl.readthedocs.io/en/latest/) – Imitation learning; Offline RL; 3D navigation; Puzzle-solving;
* [Multiagent emergence environments](https://github.com/openai/multi-agent-emergence-environments) – Multi-agent; Creating environments; Emergence behavior;
* [OpenAI Gym](https://gym.openai.com/) – Continuous control; Physics-based simulation; Classic video games; RAM state as observations;
* [OpenAI Gym Retro](https://github.com/openai/retro) – Classic video games; RAM state as observations;
* [OpenSpiel](https://github.com/deepmind/open_spiel) – Classic board games; Search and planning; Single-/Multi-agent;
* [Procgen Benchmark](https://github.com/openai/procgen) – Evaluate generalization; Procedurally-generated;
* [PyBullet Gymperium](https://github.com/benelot/pybullet-gym) – Continuous control; Physics-based simulation; MuJoCo unpaid alternative;
* [Real-World Reinforcement Learning](https://github.com/google-research/realworldrl_suite) – Continuous control; Physics-based simulation; Adversarial examples;
* [RLCard](https://github.com/datamllab/rlcard) – Classic card games; Search and planning; Single-/Multi-agent;
* [RL Unplugged](https://github.com/deepmind/deepmind-research/tree/master/rl_unplugged) – Offline RL; Imitation learning; Datasets for the common benchmarks;
* [Screeps](https://screeps.com/) – Compete with others; Sandbox; MMO for programmers;
* [Serpent.AI](https://github.com/SerpentAI/SerpentAI) – Game Agent Framework – Turn ANY video game into the RL env;
* [StarCraft II Learning Environment](https://github.com/deepmind/pysc2) – Rich action and observation spaces; Multi-agent; Multi-task;
* [The Unity Machine Learning Agents Toolkit (ML-Agents)](https://github.com/Unity-Technologies/ml-agents) – Create environments; Curriculum learning; Single-/Multi-agent; Imitation learning;
* [WordCraft](https://github.com/minqi/wordcraft) -Test core capabilities; Commonsense knowledge;

---
### [OpenAI Gym](https://gym.openai.com/)
**Ref.** [Reinforcement Learning 健身房](https://pyliaorachel.github.io/blog/tech/python/2018/06/01/openai-gym-for-reinforcement-learning.html)
![](https://i.stack.imgur.com/eoeSq.png)
1. **Agent** 藉由 action 跟 environment 互動。
2. **Environment** agent 的行動範圍，根據 agent 的 action 給予不同程度的 reward。
3. **State** 在特定時間點 agent 身處的狀態。
4. **Action** agent 藉由自身 policy 進行的動作。
5. **Reward** environment 給予 agent 所做 action 的獎勵或懲罰。

---

### Q Learning
**Blog:** [A Hands-On Introduction to Deep Q-Learning using OpenAI Gym in Python](https://www.analyticsvidhya.com/blog/2019/04/introduction-deep-q-learning-python/)<br>
![](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/04/Screenshot-2019-04-16-at-5.46.01-PM-670x440.png)
![](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/04/1_lTVHyzT3d26Bd_znaKaylQ-768x84.png)
*immediate reward r(s,a) plus the highest Q-value possible from the next state s’.* <br>
*Gamma here is the discount factor which controls the contribution of rewards further in the future.*<br>

![](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/04/Screenshot-2019-04-17-at-7.15.35-PM-768x56.png)
*Adjusting the value of gamma will diminish or increase the contribution of future rewards.*<br>

![](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/04/Screenshot-2019-03-26-at-7.57.30-PM1-768x64.png)
*where alpha is the learning rate or step size*<br>

The loss function here is mean squared error of the predicted Q-value and the target Q-value – Q*<br>

**Blog:** [An introduction to Deep Q-Learning: let’s play Doom](https://www.freecodecamp.org/news/an-introduction-to-deep-q-learning-lets-play-doom-54d02d8017d8/)<br>

<img width="50%" height="50%" src="https://cdn-media-1.freecodecamp.org/images/1*Q4XjhLC0IAOznnk5613PsQ.gif">
![](https://cdn-media-1.freecodecamp.org/images/1*js8r4Aq2ZZoiLK0mMp_ocg.png)
![](https://cdn-media-1.freecodecamp.org/images/1*LglEewHrVsuEGpBun8_KTg.png)

---
### [Gym](https://github.com/openai/gym)
Gym is an open source Python library for developing and comparing reinforcement learning algorithms by providing a standard API to communicate between learning algorithms and environments, as well as a standard set of environments compliant with that API.<br>
`pip install gym`<br>

```
import gym 
env = gym.make('CartPole-v1')

# env is created, now we can use it: 
for episode in range(10): 
    observation = env.reset()
    for step in range(50):
        action = env.action_space.sample()  # or given a custom model, action = policy(observation)
        observation, reward, done, info = env.step(action)
        if done:
            observation = env.reset()
env.close()
```
CartPole環境輸出的state包括位置、加速度、杆子垂直夾角和角加速度。

---
### [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3)
a set of reliable implementations of reinforcement learning algorithms in PyTorch.<br>
Implemented Algorithms : **A2C, DDPG, DQN, HER, PPO, SAC, TD3**.<br>
**QR-DQN, TQC, Maskable PPO** are in [SB3 Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)<br>

`pip install stable-baselines3`<br>
For Ubuntu: `pip install gym[atari]`<br>
For Win10 : `pip install --no-index -f ttps://github.com/Kojoley/atari-py/releases atari-py`<br>

**[SB3 examples](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html)**<br>

---
## DQN應用介紹
### DQN
**Paper:** [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)<br>
![](https://www.researchgate.net/publication/338248378/figure/fig3/AS:842005408141312@1577761141285/This-is-DQN-framework-for-DRL-DNN-outputs-the-Q-values-corresponding-to-all-actions.jpg)

**[PyTorch Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)**<br>
**Gym Cartpole**: [dqn.py](https://github.com/rkuo2000/RL-Gym/blob/main/cartpole/dqn.py)<br>
![](https://pytorch.org/tutorials/_images/cartpole.gif)

---
### DQN RoboCar
**Blog:** [Deep Reinforcement Learning on ESP32](https://www.hackster.io/aslamahrahiman/deep-reinforcement-learning-on-esp32-843928)<br>
**Code:** [Policy-Gradient-Network-Arduino](https://github.com/aslamahrahman/Policy-Gradient-Network-Arduino)<br>
<iframe width="482" height="271" src="https://www.youtube.com/embed/d7NcoepWlyU" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### DQN for MPPT control
**Paper:** [A Deep Reinforcement Learning-Based MPPT Control for PV Systems under Partial Shading Condition](https://www.researchgate.net/publication/341720872_A_Deep_Reinforcement_Learning-Based_MPPT_Control_for_PV_Systems_under_Partial_Shading_Condition)<br>

![](https://www.researchgate.net/publication/341720872/figure/fig1/AS:896345892196354@1590716922926/A-diagram-of-the-deep-Q-network-DQN-algorithm.ppm)

---
### DDQN 
**Paper:** [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)<br>
**Tutorial:** [Train a Mario-Playing RL Agent](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)<br>
**Code:** [MadMario](https://github.com/YuansongFeng/MadMario)<br>
```
git clone https://github.com/YuansongFeng/MadMario
cd MadMario

pip install scikit-image
pip install gym-super-mario-bros
```

Training time is around 80 hours on CPU and 20 hours on GPU.<br>
To train  : (epochs=40000)<br>
`python main.py`<br>

To replay : (modify `checkpoint = Path('trained_mario.chkpt')`)<br>
`python replay.py`<br>

![](https://pytorch.org/tutorials/_images/mario.gif)

---
### Duel DQN
**Paper:** [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)<br>
![](https://theaisummer.com/static/b0f4c8c3f3a5158b5899aa52575eaea0/95a07/DDQN.jpg)

### Double Duel Q Net
**Code:** [mattbui/dd_qnet](https://github.com/mattbui/dd_qnet)<br>

![](https://github.com/mattbui/dd_qnet/blob/master/screenshots/running.gif?raw=true)

---

## AI in Games
**Paper:** [AI in Games: Techniques, Challenges and Opportunities](https://arxiv.org/abs/2111.07631)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/AI_in_Games_survey.png?raw=true)

---
### AlphaGo
2016 年 3 月，AlphaGo 這一台 AI 思維的機器挑戰世界圍棋冠軍李世石（Lee Sedol）。比賽結果以 4 比 1 的分數，AlphaGo 壓倒性的擊倒人類世界最會下圍棋的男人。<br>
<iframe width="710" height="399" src="https://www.youtube.com/embed/1bc-8iomgB4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

**Paper:** [Mastering the game of Go with deep neural networks and tree search](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf)<br>
**Paper:** [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ)<br>

**Blog:** [Day 27 / DL x RL / 令世界驚艷的 AlphaGo](https://ithelp.ithome.com.tw/articles/10252358)<br>

AlphaGo model 主要包含三個元件：<br>
* **Policy network**：根據盤面預測下一個落點的機率。
* **Value network**：根據盤面預測最終獲勝的機率，類似預測盤面對兩方的優劣。
* **Monte Carlo tree search (MCTS)**：類似在腦中計算後面幾步棋，根據幾步之後的結果估計現在各個落點的優劣。

![](https://i.imgur.com/xdc52cv.png)

* **Policy Networks**: 給定 input state，會 output 每個 action 的機率。<br>
AlphaGo 中包含三種 policy network：<br>
* [Supervised learning (SL) policy network](https://chart.googleapis.com/chart?cht=tx&chl=p_%7B%5Csigma%7D)
* [Reinforcement learning (RL) policy network](https://chart.googleapis.com/chart?cht=tx&chl=p_%7B%5Crho%7D)
* [Rollout policy network](https://chart.googleapis.com/chart?cht=tx&chl=p_%7B%5Cpi%7D)

* **Value Network**: 預測勝率，Input 是 state，output 是勝率值。<br>
這個 network 也可以用 supervised learning 訓練，data 是歷史對局中的 state-outcome pair，loss 是 mean squared error (MSE)。

* **Monte Carlo Tree Search (MCTS)**: 結合這些 network 做 planning，決定遊戲進行時的下一步。<br>
![](https://i.imgur.com/aXdpcz6.png)
1. Selection：從 root 開始，藉由 policy network 預測下一步落點的機率，來選擇要繼續往下面哪一步計算。選擇中還要考量每個 state-action pair 出現過的次數，盡量避免重複走同一條路，以平衡 exploration 和 exploitation。重複這個步驟直到樹的深度達到 max depth L。
2. Expansion：到達 max depth 後的 leaf node sL，我們想要估計這個 node 的勝算。首先從 sL 往下 expand 一層。
3. Evaluation：每個 sL 的 child node 會開始 rollout，也就是跟著 rollout policy network 預測的 action 開始往下走一陣子，取得 outcome z。最後 child node 的勝算會是 value network 對這個 node 預測的勝率和 z 的結合。
4. Backup：sL 會根據每個 child node 的勝率更新自己的勝率，並往回 backup，讓從 root 到 sL 的每個 node 都更新勝率。

---
### AlphaZero
2017 年 10 月，AlphaGo Zero 以 100 比 0 打敗 AlphaGo。<br>
**Blog:** [AlphaGo beat the world’s best Go player. He helped engineer the program that whipped AlphaGo.](https://www.technologyreview.com/innovator/julian-schrittwieser/)<br>
**Paper:** [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)<br>
![](https://s.newtalk.tw/album/news/180/5c0f5a4489883.png)
AlphaGo 用兩個類神經網路，分別估計策略函數和價值函數。AlphaZero 用一個多輸出的類神經網路<br>
AlphaZero 的策略函數訓練方式是直接減少類神經網路與MCTS搜尋出來的πₜ之間的差距，這就是在做regression，而 AlpahGo 原本用的方式是RL演算法做 Policy gradient。(πₜ：當時MCTS後的動作機率值)<br>
**Blog:** [優拓 Paper Note ep.13: AlphaGo Zero](https://blog.yoctol.com/%E5%84%AA%E6%8B%93-paper-note-ep-13-alphago-zero-efa8d4dc538c)<br>
**Blog:** [Monte Carlo Tree Search (MCTS) in AlphaGo Zero](https://jonathan-hui.medium.com/monte-carlo-tree-search-mcts-in-alphago-zero-8a403588276a)<br>
**Blog:** [The 3 Tricks That Made AlphaGo Zero Work](https://hackernoon.com/the-3-tricks-that-made-alphago-zero-work-f3d47b6686ef)<br>
1. MTCS with intelligent lookahead search
2. Two-headed Neural Network Architecture
3. Using residual neural network architecture 
<table>
<tr>
<td><img src="https://hackernoon.com/hn-images/1*hBzorPuADtitET2SZaLN2A.png"></td>
<td><img src="https://hackernoon.com/hn-images/1*96DnPFNDD8YyN-GK737bBQ.png"></td>
<td><img src="https://hackernoon.com/hn-images/1*aJCekYFA3jG0NDBmBEYYPA.png"></td>
</tr>
</table>

![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/AlphaGo_version_comparison.png?raw=true)

---
### AlphaZero with a Learned Model
**Paper:** [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265)<br>
RL can be divided into Model-Based RL (MBRL) and Model-Free RL (MFRL). Model-based RL uses an environment model for planning, whereas model-free RL learns the optimal policy directly from interactions. Model-based RL has achieved superhuman level of performance in Chess, Go, and Shogi, where the model is given and the game requires sophisticated lookahead. However, model-free RL performs better in environments with high-dimensional observations where the model must be learned.
![](https://www.endtoend.ai/assets/blog/rl-weekly/36/muzero.png)

---

## Unsupervised Learning

### Understanding the World Through Action
**Blog:** [Understanding the World Through Action: RL as a Foundation for Scalable Self-Supervised Learning](https://medium.com/@sergey.levine/understanding-the-world-through-action-rl-as-a-foundation-for-scalable-self-supervised-learning-636e4e243001)<br>
**Paper:** [Understanding the World Through Action](https://arxiv.org/abs/2110.12543)<br>
![](https://miro.medium.com/max/1400/1*79ztJveD6kanHz9H8VY2Lg.gif)
**Actionable Models**<br>
a self-supervised real-world robotic manipulation system trained with offline RL, performing various goal-reaching tasks. Actionable Models can also serve as general pretraining that accelerates acquisition of downstream tasks specified via conventional rewards.
![](https://miro.medium.com/max/1280/1*R7-IP07Inc7K6v4i_dQ-RQ.gif)

---
## Stock RL
### Stock Price(Final Project)
**Kaggle:** [MOCI2001/stock-lstm](https://www.kaggle.com/code/moci2001/stock-lstm)<br>

**LSTM model**<br>
```
model = Sequential()
model.add(Input(shape=(history_points, 5)))
model.add(LSTM(history_points))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')
```

---
### Stock Trading
**Blog:** [Predicting Stock Prices using Reinforcement Learning (with Python Code!)](https://www.analyticsvidhya.com/blog/2020/10/reinforcement-learning-stock-price-prediction/)<br>
![](https://editor.analyticsvidhya.com/uploads/770801_26xDRHI-alvDAfcPPJJGjQ.png)

**Code:** [DQN-DDPG_Stock_Trading](https://github.com/AI4Finance-Foundation/DQN-DDPG_Stock_Trading)<br>
**Code:** [FinRL](https://github.com/AI4Finance-Foundation/FinRL)<br>
**Blog:** [Automated stock trading using Deep Reinforcement Learning with Fundamental Indicators](https://medium.com/@mariko.sawada1/automated-stock-trading-with-deep-reinforcement-learning-and-financial-data-a63286ccbe2b)<br>

**Papers:** <br>
[2010.14194](https://arxiv.org/abs/2010.14194): Learning Financial Asset-Specific Trading Rules via Deep Reinforcement Learning<br>
[2011.09607](https://arxiv.org/abs/2011.09607): FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance<br>
[2101.03867](https://arxiv.org/abs/2101.03867): A Reinforcement Learning Based Encoder-Decoder Framework for Learning Stock Trading Rules<br>
[2106.00123](https://arxiv.org/abs/2106.00123): Deep Reinforcement Learning in Quantitative Algorithmic Trading: A Review<br>
[2111.05188](https://arxiv.org/abs/2111.05188): FinRL-Podracer: High Performance and Scalable Deep Reinforcement Learning for Quantitative Finance<br>
[2112.06753](https://arxiv.org/abs/2112.06753): FinRL-Meta: A Universe of Near-Real Market Environments for Data-Driven Deep Reinforcement Learning in Quantitative Finance<br>
**Blog:** [FinRL­-Meta: A Universe of Near Real-Market En­vironments for Data­-Driven Financial Reinforcement Learning](https://medium.datadriveninvestor.com/finrl-meta-a-universe-of-near-real-market-en-vironments-for-data-driven-financial-reinforcement-e1894e1ebfbd)<br>
![](https://miro.medium.com/max/2000/1*rOW0RH56A-chy3HKaxcjNw.png)

---

## Exercises:
### Stock DQN
**Kaggle:** [MOCI2001/Stock-DQN](https://www.kaggle.com/moci2001/stock-dqn1)<br>
`cd ~/RL-gym/stock`<br>
`python train_dqn.py`<br>
![](https://github.com/MOCI2001/AI-course/blob/gh-pages/images/stock_dqn.png?raw=true)

`python enjoy_dqn.py`<br>

---
### FinRL
**Code:** [DQN-DDPG_Stock_Trading](https://github.com/AI4Finance-Foundation/DQN-DDPG_Stock_Trading)<br>
**Code:** [FinRL](https://github.com/AI4Finance-Foundation/FinRL)<br>

<br>
<br>

*This site was last updated {{ site.time | date: 01 06, 2023 }}.*
