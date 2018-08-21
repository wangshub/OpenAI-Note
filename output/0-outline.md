
# OpenAI 机器人强化学习的快速入门 **(草稿)**

> 除了试图直接去建立一个可以模拟成人大脑的程序之外， 为什么不试图建立一个可以模拟小孩大脑的程序呢?如果它接 受适当的教育，就会获得成人的大脑。 — 阿兰·图灵

## 学习目的

- 理论和仿真实践结合
- 了解掌握强化学习基本原理
- 掌握利用 Python 进行强化学习仿真

## 一. 引言介绍

强化学习 (Reinforcement learning) 是机器学习的一个子领域用于制定决策和运动自由度控制。强化学习主要研究在复杂未知的环境中，智体(agent)实现某个目标。强化学习最引人入胜的两个特点是

- **强化学习非常通用，可以用来解决需要作出一些列决策的所有问题：**例如，训练机器人跑步和弹跳，制定商品价格和库存管理，玩 Atari 游戏和棋盘游戏等等。

- **强化学习已经可以在许多复杂的环境中取得较好的实验结果：**例如 Deep RL 的 Alpha Go等

[Gym](https://gym.openai.com/docs/) 是一个研究和开发强化学习相关算法的仿真平台。

- 无需智体先验知识；
- 兼容常见的数值运算库如 TensorFlow、Theano 等

## 二. 强化学习的基本概念

强化学习也是机器学习中的一个重要分支。强化学习和监督学习的不同在 于，强化学习问题不需要给出“正确”策略作为监督信息，只需要给出策略的(延迟)回报，并通过调整策略来取得最大化的期望回报。

### 2.1 术语

- 智体 (Agent)
- 环境 (Environment)
- 状态 (State)
- 动作 (Action)
- 策略 (Policy)
- 奖励 (Reward)
- 状态转移概率 

### 2.2 马尔科夫过决策过程

- 图片
- 解释

## 三. OpenAI 强化学习仿真环境

### 3.1 环境安装

- step1
- step2
- step3

### 3.2 OpenAI 术语解释

- Observations
- Spaces
- Loop
- Render

## 四. 第一个强化学习 Hello World


```python
print('Hello reinforcement learning !\n'*3)
```

    Hello reinforcement learning !
    Hello reinforcement learning !
    Hello reinforcement learning !
    


## 五. OpenAI 强化学习进阶


```python
print('Hello reinforcement learning !\n'*4)
```

    Hello reinforcement learning !
    Hello reinforcement learning !
    Hello reinforcement learning !
    Hello reinforcement learning !
    


## 六. 总结与扩展

- 项目地址
- 扩展阅读文献 1
- 扩展阅读文献 2

