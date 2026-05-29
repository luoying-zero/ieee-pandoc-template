# 分布式深度强化学习技术综述：架构、并行化与扩展策略

## 摘要

深度强化学习（DRL）在游戏博弈、机器人控制和自主决策等领域取得了突破性进展，但其训练过程需要大量的环境交互数据和神经网络模型参数，导致计算密集且耗时严重。分布式深度强化学习（DDRL）通过并行和分布式计算技术有效解决了这一瓶颈，使得大规模DRL系统的训练成为可能。本文系统综述了分布式强化学习的核心技术，涵盖系统架构设计、仿真并行化、计算并行化、分布式同步机制、进化强化学习以及数据、网络和训练预算的扩展策略。通过对代表性方法和开源工具的深入分析，本文旨在为研究人员提供分布式DRL领域的全面技术图谱，并指出未来的研究方向。

**关键词**：深度强化学习，分布式训练，并行计算，数据扩展，网络扩展，训练预算扩展

---

## 1. 引言

深度强化学习自Deep Q-Network（DQN）在Atari游戏上实现人类水平控制以来，已成为解决序列决策问题的主流范式[1]。然而，DRL的试错学习机制需要智能体与环境进行海量交互以收集经验数据，同时深度神经网络的前向推理和反向传播计算也极为耗时。例如，DQN在Atari 2600游戏上训练约需50M帧的经验，耗时约38天；而更复杂的应用如OpenAI Five（Dota 2）和AlphaStar（星际争霸II）则需要数十万CPU核心和数百个GPU/TPU进行数周至数月的训练[2][3][4]。

为应对上述挑战，分布式深度强化学习（DDRL）应运而生。其核心思想是将DRL训练过程分解为数据收集（采样）和模型更新（学习）两个主要阶段，通过并行化和分布式计算技术加速整个训练流程[5][6]。近年来，研究者从多个维度探索了DRL的扩展问题：（1）**数据扩展**——通过并行数据采集和合成数据生成扩大训练数据规模；（2）**网络扩展**——通过增加网络宽度、深度、使用集成方法或混合专家模型提升模型容量；（3）**训练预算扩展**——通过分布式训练、高重放比率、大批量训练和辅助训练优化计算资源利用[7]。

本文综合上述视角，系统梳理分布式深度强化学习的技术体系，重点讨论系统架构、并行化策略、同步机制和扩展方法，并展望未来发展方向。

---

## 2. 分布式强化学习的系统架构

分布式DRL系统的核心组件可抽象为四个部分[5][8]：

- **Actor（执行器）**：负责与环境交互，根据策略网络生成动作并收集经验数据（状态、动作、奖励、下一状态）。
- **Learner（学习器）**：负责基于经验数据计算梯度并更新神经网络参数。
- **Parameter Server（参数服务器）**：维护全局模型的最新参数，供Actor和Learner同步。
- **Replay Memory（经验回放缓冲区）**：存储Actor收集的经验数据（主要用于离策略方法）。

根据这些组件的组织方式，DDRL架构可分为两大类：**中心化架构**和**去中心化架构**[5][6]。

### 2.1 中心化架构

中心化架构中，存在一个中心节点维护全局模型。Gorila（General Reinforcement Learning Architecture）是最早的大规模分布式DRL框架之一[9]，它包含参数服务器、多个Actor和多个Learner。Actor将经验数据发送至Replay Memory，Learner从中采样计算梯度并提交至参数服务器更新全局模型。这种星型通信拓扑使得参数同步简单高效，但参数服务器可能成为通信瓶颈。

APE-X[10]在Gorila基础上引入优先经验回放，仅用一个GPU Learner维护最新参数替代传统参数服务器，通过优先采样TD误差较大的经验提高学习效率。R2D2[11]进一步引入LSTM循环神经网络处理部分可观测环境，在APE-X架构下实现了Atari游戏中52/57款的超人类水平。

A3C（Asynchronous Advantage Actor-Critic）[12]提出了一种基于CPU多线程的异步中心化架构。每个线程独立完成Actor和Learner的工作，与环境交互并计算梯度异步更新全局网络。A3C使用16个CPU核心即可在Atari任务上达到优于GPU版DQN的性能，且训练时间减半。

### 2.2 去中心化架构

去中心化架构消除了中心节点，采用多Learner并通过All-reduce通信机制聚合梯度[5][13]。IMPALA[14]是代表性的去中心化架构，Actor使用CPU核心进行环境交互，Learner部署在GPU上执行梯度计算。多个Learner之间同步聚合梯度，Actor从Learner获取最新参数。IMPALA引入V-trace离策略修正算法处理Actor与Learner之间的策略延迟问题，可在DMLab-30和Atari-57上实现高吞吐量训练。

DD-PPO[15]将去中心化架构扩展到多机器场景，每个工作节点封装Actor和Learner功能，交替执行经验收集、梯度聚合和参数优化。使用128个GPU可实现107倍的加速比。SEED RL[13]进一步优化通信层，通过gRPC高性能RPC库实现Actor与Learner之间仅传输状态和动作的轻量级通信，在Atari-57、DeepMind Lab和Google Research Football等环境上取得了40%-80%的成本降低。

### 2.3 混合架构与新兴趋势

为解决中心化架构的瓶颈问题和去中心化架构的同步开销，研究者提出了多种创新架构。Gossip-based架构[16]采用点对点通信拓扑，每个工作节点仅与邻居通信，理论保证模型在训练过程中保持在ε-close范围内。SampleFactory[17]在单机环境下通过异步强化学习获得130K FPS的吞吐量，比基线SEED RL快4倍。SRL[18]将训练任务细粒度分解为Rollout worker、Policy worker和Learner，实现在超过一万个核心上的高效扩展。

---

## 3. 仿真并行化

DRL训练需要与仿真环境（如OpenAI Gym、MuJoCo、Unity ML等）进行大量交互以收集样本。当训练复杂应用（如PointGoal导航需要25亿帧经验[15]）时，仿真效率成为关键瓶颈。现有的仿真并行化策略主要分为两类[6]。

### 3.1 基于CPU集群的分布式仿真

传统的仿真并行化方法是在CPU集群上部署大量环境实例并行运行[5][19]。每个Actor（通常运行在一个进程或线程中）管理一个环境实例，通过随机策略或不同探索策略与环境交互产生多样化经验。APE-X使用360个Actor机器实现了约50K环境帧/秒（FPS）的数据生成速率[10]，IMPALA使用500个CPU核心和8个P100 GPU可达到250K FPS[14]。

然而，CPU集群扩展存在若干挑战：（1）随CPU核心和节点数量增加，节点间通信、同步和资源分配开销增大[15][20]；（2）在CPU-GPU混合架构中，经验数据需要频繁在CPU内存和GPU内存之间拷贝，引入额外的上下文切换开销[6]。

### 3.2 基于GPU/TPU的批量仿真

为解决CPU仿真的扩展限制和通信瓶颈，研究者转向利用GPU/TPU等专用硬件进行批量仿真。Liang等人[21]提出GPU加速的RL模拟器，在单机单GPU单CPU配置下可并行仿真数千个人形机器人并生成60K FPS。

零拷贝批量仿真是近期的重要研究方向。Isaac Gym[22]提供Tensor API直接访问GPU缓冲区中的仿真结果，所有计算保留在GPU内，消除了CPU-GPU间的片外通信瓶颈，在单个GPU上实现数千个环境的并行仿真，训练时间比之前的工作提高300倍。Brax[23]基于TPU实现了MuJoCo-ant任务上每秒数亿步的仿真速度。CuLE[24]支持在GPU内存中直接运行Atari仿真，使用1个GPU并行4096个环境可达155K FPS。

批量仿真的关键优化技术包括：（1）共享场景资源，减少GPU内存占用；（2）摊销仿真计算、同步和通信成本；（3）利用张量操作的大规模并行性。

---

## 4. 计算并行化

除仿真外，DRL训练还涉及神经网络推理、反向传播和进化计算等密集计算任务。计算并行化技术主要包括三类[6]。

### 4.1 集群计算

集群计算通过高速局域网互联的多台机器协同完成计算任务。Gorila[9]是早期将集群计算引入DRL的代表性工作，使用31台机器实现相比单机GPU实现10倍的速度提升。后续的IMPALA[14]、R2D2[11]等方法普遍采用上百个CPU核心和数个到数十个GPU的集群配置。AlphaStar[2]在训练中使用3072个TPU v3和50400个CPU核心，OpenAI Five[4]部署了1536个GPU和172800个CPU核心。这些大规模系统展示了集群计算在DRL训练中的强大能力。

### 4.2 单机并行化

单机并行化利用多处理器或多核CPU在单台工作站中提高计算能力，避免了跨机器通信和同步的开销[25]。A3C[12]使用16个CPU核心的多线程并行，性能优于单GPU实现。SampleFactory[17]在36个CPU核心和一块RTX 2080Ti GPU的单机配置下实现了4倍于SEED RL的加速。

单机并行化的优势在于硬件门槛低、通信开销小，适合学术研究和小规模实验。通过优化资源利用率和调度策略（如GA3C[26]使用CPU/GPU混合架构，GPU负责训练和推理，CPU负责环境交互），单机方案能够提供可观的计算吞吐量。

### 4.3 专用硬件架构加速

GPU的大规模并行计算能力使其成为神经网络训练的首选硬件。IMPALA[14]使用8个P100 GPU提升多任务DRL训练速度，rlpyt[27]利用8个P100 GPU获得6倍加速比。NVIDIA的Tensor Core等专用计算单元进一步增强了GPU的神经网络计算性能。

FPGA（现场可编程门阵列）在降低能耗方面具有优势。FA3C[28]基于FPGA实现了比GPU（Tesla P100）高27.9%的推理吞吐量和1.62倍的能效比。PPO_FPGA[29]在Xilinx Alveo U200上实现相比Titan Xp GPU的27.5倍加速。基于CPU-FPGA异构平台的片上重放管理[30]相比GPU（GTX 3090）提升4.3倍IPS。

TPU（张量处理单元）是Google专为机器学习设计的ASIC。AlphaZero[3]使用5000个TPU v1和64个TPU v2核心在24小时内完成训练，SEED RL[13]使用TPU v3实现了比IMPALA快11倍的训练速度。

### 4.4 计算任务调度

在异构计算环境中，有效的任务调度对分布式DRL系统效率至关重要[6]。现有的调度策略包括：

- **负载均衡**：Ray[31]使用细粒度和粗粒度负载均衡方法，rlpyt[27]形成两组交替的仿真进程组交替服务GPU以保持高利用率。
- **资源动态调整**：MINIONSRL[32]动态调整Actor数量以最小化训练时间和成本。
- **计算与通信重叠**：PEARL[33]设计支持计算与通信重叠调度的Learner模块。
- **抢占式调度**：DD-PPO[15]引入慢速节点预抢占机制，当一定比例的工作节点完成数据收集后终止滞后节点。

---

## 5. 分布式同步机制

在数据并行分布式训练中，同步机制对训练效率和模型质量有重要影响。根据DRL的离策略/在策略特性，分布式同步机制可分为两类[5][6]。

### 5.1 异步离策略训练

异步训练中，各个工作节点独立运行，按照各自的节奏更新模型，无需等待其他节点完成[9][10][11]。这种方式天然导致Actor端的行为策略模型和Learner端的目标策略模型之间存在差异，因此主要适用于离策略算法（如DQN系列），这些算法不要求行为策略与目标策略严格一致。

然而，异步训练面临两大挑战[5][14]：
1. **策略延迟（Policy Lag）**：目标策略分布与行为策略收集的训练数据分布之间的偏移可能导致不稳定。IMPALA提出的V-trace离策略修正算法通过截断重要性采样权重改善收敛性[14]，GA3C引入ε-修正防止策略概率的对数值过小[26]。
2. **陈旧更新（Stale Update）**：工作节点使用过时版本的模型生成训练数据更新最新模型，可能减缓收敛速度。

### 5.2 同步在策略训练

为解决异步训练的稳定性问题，同步训练要求所有工作节点在每个迭代中等待梯度计算完成后再同步更新模型[5]。这种方式保证了各节点模型参数的一致性，满足在策略算法对行为策略与目标策略一致性的要求。

同步训练可通过两种方式实现[5][14][15]：
- **中心化同步**：如PAAC[34]和DBA3C[35]，各工作节点传输梯度至参数服务器，全局模型更新后广播至所有节点。
- **去中心化同步**：如DD-PPO[15]，各节点梯度经All-reduce通信聚合后在本地更新模型。

同步训练的主要缺点是同步屏障（Synchronization Barrier）——快节点需等待慢节点完成，导致计算资源利用不充分，尤其在异构集群环境中更为突出[15]。

### 5.3 混合策略与未来方向

为进一步平衡效率和稳定性，近期研究探索了多种混合策略：

- **陈旧同步训练（Stale Synchronous Parallel, SSP）**：允许快节点在有限陈旧度内异步进行，当陈旧度达到阈值时强制执行同步更新[6]。虽然SSP在深度学习中已展现良好效果，但在分布式DRL中应用尚不充分，具有重要研究价值。
- **离策略与在策略混合**：Schmitt等人[36]将离策略回放经验与在策略数据混合训练，利用信任区域算法缓解偏差，在分布式设置中优于基于V-trace的IMPALA。
- **基于学习的同步决策**：AutoSync[37]通过学习动态决定同步时机，自适应优化资源利用和模型一致性之间的权衡。

---

## 6. 深度进化强化学习

进化计算作为有别于梯度下降的优化方法，在DRL训练中展现了独特优势[38][39]。进化方法直接在参数空间搜索，通过进化和变异候选解种群，无需反向传播梯度，天然支持大规模并行化且通信带宽需求低[6][40]。

### 6.1 基于进化策略的加速

进化策略（Evolution Strategies, ES）通过估计目标函数在参数空间上的自然梯度来更新搜索分布[41]。Salimans等人[42]将ES应用于DRL，将噪声扰动后策略的期望回报作为优化目标，使用蒙特卡洛估计近似自然梯度。ES在参数空间而非动作空间探索，对动作频率和延迟奖励不敏感，特别适合长时间范围问题。

ES的可扩展性优势体现在[42]：每次迭代操作在完整episode上进行，通信频率大幅降低；传输信息仅限episode回报标量，带宽需求远低于梯度向量传输；无需价值函数近似器的额外梯度同步。在80台机器1440个CPU核心的集群上，ES实现了相比单机两个数量级的训练时间缩减。

### 6.2 基于遗传算法的加速

遗传算法（Genetic Algorithms, GA）是梯度无关的进化方法[43]。Deep GA[38]通过截断选择、高斯噪声变异等操作进化神经网络参数，结合压缩编码技术（使用随机种子列表重建参数向量）支持大规模分布式部署。实验表明，Deep GA可超越ES、A3C和DQN的平均性能，且在解决局部最优问题上优于梯度方法，能够跨越参数空间中的局部谷底。

NEAT[44]及其变体进一步进化网络拓扑结构，不仅优化权重还在演化过程中搜索网络架构。尽管早期工作针对的是小型网络，近期研究已扩展至深度网络[45][46][47]和超参数优化[48]，通过自动化网络架构设计减轻手动调参负担。

### 6.3 进化与梯度学习的融合

进化计算和梯度下降各有优势，融合两种方法正成为重要研究方向[49]。进化引导策略梯度（Evolution-Guided Policy Gradient）[50]将遗传算法与DDPG结合：维护Actor网络种群与环境交互产生多样化经验，通过选择、变异、交叉创造下一代Actor；Critic网络和Actor网络基于经验回放缓冲区进行梯度学习，学习行为注入进化种群。

遗传策略优化（Genetic Policy Optimization, GPO）[51]在状态访问空间进行交叉（克隆双亲行为）而非参数空间交叉，利用PPO算法进行变异而非随机扰动，在MuJoCo基准上优于PPO和A2C。

进化强化学习方法在复杂形态学控制[45]等任务中展示了巨大潜力，通过进化多样化智能体形态协同学习运动和操作技能，使用1152个CPU进化10代种群并训练4000个智能体形态，每个形态经历500万次环境交互。

---

## 7. 数据、网络与训练预算的扩展策略

结合Ma等人[7]的系统化分类，分布式DRL的扩展策略可从数据扩展、网络扩展和训练预算扩展三个维度展开分析。

### 7.1 数据扩展

数据扩展通过增加训练数据的数量和质量来提升DRL性能，主要包括并行数据收集和合成数据生成两类方法[7]。

**并行数据收集**通过多工作节点并行与环境交互显著提高数据采集效率。Ape-X[10]通过解耦Actor（负责并行环境交互和多样化探索）和Learner（负责集中式优先回放学习），证明了并行化对学习速度和最终策略质量的显著提升。SAPG[52]将大规模并行环境划分为多个块，每块由不同策略管理，通过重要性采样聚合数据，结合集成多样化探索和熵正则化，在复杂机器人操控任务中超越PPO和Population-based训练。

在离线RL中，ExORL[53]利用无监督无奖励探索收集多样化单任务或多任务数据集，然后为下游任务重新标记，证明了数据多样性和规模对有效离线策略训练的关键作用。Scaled QL[54]使用ResNet架构和分布交叉熵损失，在40个Atari游戏上训练单个策略（8000万参数），仅使用51%数据集即达到人类水平表现，展示了模型容量和数据规模的协同效应。

**合成数据生成**通过生成模型扩充训练数据集。BooT[55]利用Transformer序列模型生成合成轨迹，通过自回归或教师强制模式生成高置信度自举数据，在D4RL基准上实现了优越性能。SYNTHER[56]使用扩散模型生成合成转换（synthetic transitions）增强有限真实经验，在离线RL中有效训练更大的策略/价值网络，并在在线RL中通过高UTD比率提升样本效率。PGR[57]集成条件生成模型与相关性引导合成数据生成，训练扩散模型生成基于好奇心或价值函数优化的过渡样本，相比SYNTHER进一步提升了样本效率和可扩展性。

### 7.2 网络扩展

网络扩展通过增加模型容量来提升DRL系统的表达能力和性能，主要包括网络规模扩展、集成方法和智能体数量扩展[7]。

**网络规模扩展**涉及增加层宽和层深。OFENet[58]探索增加输入维度对DRL的影响，通过辅助预测任务学习高维状态表示，挑战了低维状态固有更有效的传统假设。CrossQ[59]利用批重归一化稳定无需目标网络的训练，并采用更宽的Critic层改善优化。SimBa[60]嵌入简单性偏差（观测归一化、残差前馈块和后层归一化），在DMC、MyoSuite、HumanoidBench等基准上匹配或超越强基线BRO。

BRO[61]结合正则化Critic网络扩展与层归一化、权重衰减和乐观探索等技术，在40个复杂任务中实现高样本效率，证明了Critic网络战略扩展在复杂领域（如肌肉骨骼控制和人形机器人移动）中的优势。BBF[62]通过系统扩展ResNet架构的网络宽度，结合周期性参数重置、退火更新视野、增加折扣因子和权重衰减等技术，在Atari 100K基准上实现超人类水平，表明网络大小扩展需要与样本高效训练机制仔细协同设计。

**Transformer架构**为跨任务泛化提供了独特优势。Gato[63]基于12亿参数的单个Transformer模型处理600多项任务，包括机器人控制、Atari游戏和图像描述，展示了网络扩展实现跨域泛化的潜力。GTrXL[64]在Transformer中引入门控机制和修改的层归一化顺序，稳定训练12层网络、512步记忆，在DMLab-30等记忆密集型RL基准上达到最优性能。PAC[65]证明了离线Actor-Critic RL也能有效扩展至大型Transformer模型，遵循类似于监督学习的扩展规律。

**深度扩展**是近期重要突破。DT-VINs[66]通过动态转换核和自适应高速连接损失（adaptive highway loss）实现高达5000层的超深度架构，在100×100迷宫导航和需要1800+步规划的3D ViZDoom环境中实现最优性能。Wang等人[67]证明将网络深度扩展至1024层可显著增强目标条件任务中自监督RL的性能，在移动和操作环境中实现2-50倍提升，深层网络涌现出复杂迷宫导航和人形机器人杂技等新能力。

**集成方法**通过聚合多个模型提升鲁棒性和稳定性。Bootstrapped DQN[68]使用多头Q网络（共享卷积层）+ Bootstrap采样近似后验分布，实现了类似Thompson采样的深度探索。REDQ[69]通过随机集成子采样（Randomized Ensembled Double Q-Learning）实现高UTD比率，在MuJoCo基准上达到与基于模型方法相当的样本效率。DroQ[70]使用小集成Q函数+ Dropout连接+层归一化，在保持样本效率的同时提升计算效率。

Maxmin Q-learning[71]利用N个动作价值估计的最小值来控制估计偏差（从高估到低估），理论分析证明在最优N选择下可实现无偏估计且方差低于标准Q-learning。TQC[72]将分布RL、截断和集成结合，通过截断N个分布Critic的分位数估计右尾并平均保留的原子实现比最小集成方法更精细的偏差控制。

**混合专家模型（MoE）**代表了集成扩展的结构化创新。Obando-Ceron等人[73]将Soft MoE集成到DQN和Rainbow等价值基础RL架构中，在Atari基准上展示随专家数量增加而显著性能提升的效果，与传统参数扩展方法性能下降形成对比。

**智能体数量扩展**主要通过进化强化学习（ERL）实现[74]。ERL框架由三个核心组件构成：进化种群（经进化策略向更优候选者进化）、RL智能体（按RL算法学习）和交互模块（控制种群与RL智能体的交互）[7][49]。ERL[50]将DDPG和遗传算法结合，种群中高适应度策略成为精英并通过变异和交叉操作渐进创造下一代Actor，种群数据和RL智能体定期注入相互作用。

CERL[75]通过使用多个具有不同折扣因子的RL智能体进一步强化探索，在人形机器人任务中显著优于ERL。ERL-Re2[76]发现独立策略架构导致冗余学习问题，提出将策略分解为共享状态表示和独立线性策略表示，在更大种群规模下实现高效知识转移。EvoRainbow[77]系统比较了现有ERL方法在交互模块、个体架构等方面的设计选择，识别出最佳组合。

### 7.3 训练预算扩展

训练预算扩展通过优化计算资源分配来提升训练效率，主要包括分布式训练、重放比率扩展、批量大小扩展和辅助训练[7]。

**分布式训练**通过系统架构优化最大化硬件利用率和数据吞吐量。IMPALA[14]的解耦Actor-Learner架构和V-trace离策略修正使训练能扩展到数千台机器（250K FPS）。QT-Opt[78]使用分布式离策略Q-learning方法基于580,000+次真实抓取尝试训练视觉机器人操作，达到96%成功率，证明大规模数据对学习鲁棒闭环策略的重要性。

**重放比率扩展**（UTD比率）通过增加每采样经验的梯度更新次数来最大化数据利用率，但面临估计偏差放大、灾难性过拟合和网络可塑性损失等挑战[7][79]。SR-SAC/SR-SPR[80]通过周期性地重置网络参数缓解可塑性损失，使UTD比率扩展至128倍，在Atari 100k和DMC上实现优越性能。PLASTIC[81]协同结合锐度感知优化、层归一化、周期参数重置和CReLU激活，在UTD=8时保持输入可塑性和标签可塑性，实现样本高效和计算帕累托最优。

AVTD[82]通过验证TD误差识别回放缓冲区过拟合为主要瓶颈，训练多个采用不同正则化策略的智能体，动态选择验证TD误差最低的智能体进行环境交互。MAD-TD[83]通过学到的世界模型进行模型增强数据稳定，仅用5%模型生成状态-动作对正则化价值估计，在UTD=16时实现稳定训练。MARR[84]引入Shrink & Perturb策略周期性地重置网络参数，在并行环境设置下将多智能体重放比率扩展至50。

**批量大小扩展**通过大规模并行优化来稳定梯度估计和加速收敛[7]。LaBER[85]通过大预采样批次、代理优先级计算和下采样来近似梯度范数分布，在Atari游戏和连续控制任务上比优先经验回放（PER）和均匀采样收敛更快。在离线RL中，Nikulin等人[86]利用大批量优化加速Q集成方法，通过平方根学习率调整和批量扩展替代集成扩展，将训练时间减少3-4倍。

**辅助训练**通过在常规RL目标外注入额外信号来学习有用表示和促进策略学习[7]。SAC+AE[87]结合VAE重构辅助任务提取紧凑视觉观测表示，显著提升SAC在DMC视觉运动任务中的学习效率。CURL[88]将对比学习融入SAC，使用InfoNCE损失和MoCo架构学习潜在表示空间，促进相似状态泛化。

UNREAL[89]引入奖励预测、像素变化最大化和额外离策略价值网络训练等辅助训练目标，在Labyrinth 3D视觉观测游戏和Atari游戏上显著改善A3C的收敛性能和样本效率。SPR[90]通过最小化真未来状态与预测未来状态在表示空间中的预测误差（借助多步动态转换模型），学习预测环境动态的表示，在Atari任务上显著提升性能和样本效率。

---

## 8. 开源库与平台

为促进分布式DRL算法的开发和实验，研究者开发了多个开源库和平台[5][6]。下表比较了代表性工具的主要特征：

| 库/平台 | 机构 | 年份 | 主要DRL算法 | 支持环境 | 分布式特性 |
|---------|------|------|------------|---------|-----------|
| Ray RLlib[31] | UC Berkeley | 2017 | A2C/A3C/PPO/IMPALA/APEX/DDPG/SAC等 | Gym/PettingZoo/Unity3D | 同步/异步训练，落后节点迁移 |
| Acme[91] | DeepMind | 2020 | D4PG/MPO/IMPALA/R2D2/MCTS等 | DMC/Gym | 异步训练，低层存储系统Reverb |
| SEED RL[13] | Google | 2020 | IMPALA/R2D2/SAC/PPO/V-MPO等 | Atari/DMC/Google Football | 中心化推理，gRPC快速通信 |
| SampleFactory[17] | Intel Lab | 2020 | PPO | Atari/MuJoCo/ViZDoom/IsaacGym等 | 异步训练，单机优化 |
| rlpyt[27] | UC Berkeley | 2019 | A2C/PPO/DQN/R2D2/SAC等 | Atari/Gym | 同步/异步采样和优化 |
| Tianshou[92] | 清华 | 2022 | PPO/DDPG/TD3/SAC/DQN/Rainbow等 | Gym/PettingZoo | 同步/异步训练，标准化训练流程 |
| PARL[93] | 百度 | 2019 | DQN/DDPG/PPO/IMPALA/SAC/MADDPG等 | Gym | 中心化架构，数千CPU和多GPU并行 |
| Fiber[94] | Uber | 2020 | A3C/PPO/ES等 | ALE/Gym/MuJoCo | 标准多进程API，在线迁移 |
| TorchRL[95] | UPF | 2023 | A2C/PPO/SAC/REDQ/IMPALA/APEX等 | Gym/DMC | 高模块化，灵活组合 |

这些平台提供了丰富的基线算法和环境集成，支持从单机多进程到多机器集群的各种部署场景，极大降低了分布式DRL开发和实验的门槛。

---

## 9. 未来方向与挑战

尽管分布式DRL取得了显著进展，但仍面临多项挑战和机遇[6][7]。

### 9.1 扩展维度间的交互关系优化

当前研究通常将数据扩展、网络扩展和训练预算扩展视为独立维度，但其相互依赖关系尚不清晰[7]。合成数据的有效性取决于策略网络的能力，最优UTD比率随Critic宽度非线性变化。开发自适应框架（如通过元学习或可微架构搜索）联合优化这些维度对于理解不同扩展策略的协同效应至关重要。

### 9.2 专用硬件加速器设计

设计针对神经网络计算高效的专用硬件架构是重要方向[6]。内存计算（in-memory computing）可减少数据移动和延迟，适合DRL训练。流水线并行化DRL训练负载的硬件设计可确保训练过程中硬件阵列始终保持活跃状态。神经形态芯片（如基于脉冲的神经形态芯片）在提升计算效率同时降低能耗具有潜力。异构架构结合不同硬件平台优势值得进一步探索。

### 9.3 网内分布式聚合机制

网络通信在分布式梯度聚合中占据大量执行时间[6]。将梯度聚合过程从工作节点转移到可编程交换机等网络设施本身的网内计算（in-switch computing）正成为新兴趋势。这种方法在网络数据包粒度而非工作节点内存中的梯度向量粒度进行聚合，可大幅降低端到端网络延迟和传输数据量。

### 9.4 高效样本利用算法

大量模拟数据需求是DRL训练的已知问题之一，部分原因是样本复用率低[4]。开发能更有效利用模拟样本而不损害学习稳定性的算法至关重要。更精密的回放机制（如Priority-refresh[96]、异步课程经验回放[97]和遗憾最小化回放[98]），以及在大型状态-动作空间中的探索策略（如好奇心驱动[99]、多样性驱动[100]和新奇性搜索[101]）都是有前景的方向。

### 9.5 大语言模型增强的DRL

大语言模型（LLMs）装备大规模预训练知识和强大泛化能力，可为DRL的奖励函数设计、动作选择、策略评估等方面提供有力支持[6][102]。基于LLM的建模能力和常识预训练知识，不仅可在训练开始时赋予相当的能力水平，还可辅助优化过程中的决策制定。这一方向已涌现许多先驱工作，预计将成为稳健发展的活跃领域。

### 9.6 可扩展性与效率悖论

DRL独特的非平稳目标和延迟信用分配使得网络扩展规律虽与监督学习相似，却面临更深层次挑战[7]。1000层网络表现出的涌现能力表明深度扩展仍有巨大潜力，但当前架构在奖励信号下扩展深度仍面临困难，且缺乏指导深度/宽度配置的统一理论。建立连接表示能力、优化景观几何和探索-利用权衡的容量扩展统一理论仍是开放挑战。

### 9.7 基准测试和评估体系

当前的扩展研究集中在狭窄的任务分布上（如DMC连续控制），忽视了对开放世界复杂性的泛化评估[7]。亟需开发标准化的基准测试来评估：（1）跨域迁移，（2）组合推理，（3）对分布偏移的鲁棒性。此外，社区缺乏一致的扩展效率指标——应将样本复杂度、计算成本和能源消耗纳入多目标评估体系。

---

## 10. 结论

分布式深度强化学习通过并行和分布式计算技术有效解决了DRL训练的计算密集型挑战，已成为训练大规模智能体的关键技术。本文从系统架构、仿真并行化、计算并行化、同步机制、进化强化学习和数据/网络/训练预算扩展策略等多个维度进行了全面综述。当前，中心化和去中心化架构并存且不断演化，异步和同步机制各有适用场景，进化计算与梯度学习相互融合，扩展策略从单一维度走向多维度协同优化。随着专用硬件加速、网内聚合、高效样本利用和大语言模型增强等新兴方向的推进，分布式DRL将继续向更高效、更可扩展、更通用的方向发展，为复杂决策问题的解决提供强大支撑。

---

## 参考文献

[1] V. Mnih et al., "Human-level control through deep reinforcement learning," Nature, vol. 518, pp. 529-533, 2015.

[2] O. Vinyals et al., "Grandmaster level in StarCraft II using multi-agent reinforcement learning," Nature, vol. 575, pp. 350-354, 2019.

[3] D. Silver et al., "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play," Science, vol. 362, pp. 1140-1144, 2018.

[4] C. Berner et al., "Dota 2 with large scale deep reinforcement learning," arXiv:1912.06680, 2019.

[5] Q. Yin et al., "Distributed deep reinforcement learning: A survey and a multi-player multi-agent learning toolbox," Machine Intelligence Research, vol. 21, no. 3, pp. 411-430, 2024.

[6] Z. Liu et al., "Acceleration for deep reinforcement learning using parallel and distributed computing: A survey," arXiv:2411.05614, 2024.

[7] Y. Ma et al., "Scaling DRL for decision making: A survey on data, network, and training budget strategies," arXiv:2508.03194, 2025.

[8] A. Nair et al., "Massively parallel methods for deep reinforcement learning," ICML Workshop, 2015.

[9] A. Nair et al., "Massively parallel methods for deep reinforcement learning," arXiv:1507.04296, 2015.

[10] D. Horgan et al., "Distributed prioritized experience replay," ICLR, 2018.

[11] S. Kapturowski et al., "Recurrent experience replay in distributed reinforcement learning," ICLR, 2019.

[12] V. Mnih et al., "Asynchronous methods for deep reinforcement learning," ICML, pp. 1928-1937, 2016.

[13] L. Espeholt et al., "Seed RL: Scalable and efficient deep-RL with accelerated central inference," ICLR, 2020.

[14] L. Espeholt et al., "IMPALA: Scalable distributed deep-RL with importance weighted actor-learner architectures," ICML, pp. 2263-2284, 2018.

[15] E. Wijmans et al., "DD-PPO: Learning near-perfect PointGoal navigators from 2.5 billion frames," ICLR, 2020.

[16] M. Assran et al., "Gossip-based actor-learner architectures for deep reinforcement learning," NeurIPS, 2019.

[17] A. Petrenko et al., "Sample factory: Egocentric 3D control from pixels at 100000 FPS with asynchronous reinforcement learning," ICML, pp. 7608-7618, 2020.

[18] Z. Mei et al., "SRL: Scaling distributed reinforcement learning to over ten thousand cores," ICLR, 2024.

[19] E. Liang et al., "RLlib: Abstractions for distributed reinforcement learning," ICML, pp. 3053-3062, 2018.

[20] R. Mittal et al., "TIMELY: RTT-based congestion control for the datacenter," ACM SIGCOMM, 2015.

[21] J. Liang et al., "GPU-accelerated robotic simulation for distributed reinforcement learning," CoRL, 2018.

[22] V. Makoviychuk et al., "Isaac Gym: High performance GPU-based physics simulation for robot learning," arXiv:2108.10470, 2021.

[23] C. D. Freeman et al., "Brax – A differentiable physics engine for large scale rigid body simulation," arXiv:2106.13281, 2021.

[24] S. Dalton and I. Frosio, "Accelerating reinforcement learning through GPU Atari emulation," NeurIPS, 2020.

[25] A. Stooke and P. Abbeel, "Accelerated methods for deep reinforcement learning," arXiv:1803.02811, 2018.

[26] M. Babaeizadeh et al., "Reinforcement learning through asynchronous advantage actor-critic on a GPU," ICLR, 2017.

[27] A. Stooke and P. Abbeel, "rlpyt: A research code base for deep reinforcement learning in PyTorch," 2019.

[28] H. Cho et al., "FA3C: FPGA-accelerated deep reinforcement learning," ASPLOS, pp. 499-513, 2019.

[29] Y. Meng et al., "Accelerating proximal policy optimization on CPU-FPGA heterogeneous platforms," FCCM, pp. 19-27, 2020.

[30] Y. Meng et al., "FPGA acceleration of deep reinforcement learning using on-chip replay management," FPGA, pp. 40-48, 2022.

[31] P. Moritz et al., "Ray: A distributed framework for emerging AI applications," OSDI, 2018.

[32] H. Yu et al., "Cheaper and faster: Distributed deep reinforcement learning with serverless computing," AAAI, pp. 16539-16547, 2024.

[33] Y. Meng et al., "PEARL: Enabling portable, productive, and high-performance deep reinforcement learning using heterogeneous platforms," CF, pp. 41-50, 2024.

[34] A. V. Clemente et al., "Efficient parallel methods for deep reinforcement learning," arXiv:1705.04862, 2017.

[35] I. Adamski et al., "Distributed deep reinforcement learning: Learn how to play Atari games in 21 minutes," ISC, pp. 370-388, 2018.

[36] S. Schmitt et al., "Off-policy actor-critic with shared experience replay," ICML, pp. 8545-8554, 2020.

[37] H. Zhang et al., "AutoSync: Learning to synchronize for data-parallel distributed deep learning," NeurIPS, pp. 906-917, 2020.

[38] F. P. Such et al., "Deep neuroevolution: Genetic algorithms are a competitive alternative for training deep neural networks for reinforcement learning," arXiv:1712.06567, 2017.

[39] K. O. Stanley et al., "Designing neural networks through neuroevolution," Nature Machine Intelligence, vol. 1, no. 1, pp. 24-35, 2019.

[40] A. Gupta et al., "Embodied intelligence via learning and evolution," Nature Communications, 2021.

[41] D. Wierstra et al., "Natural evolution strategies," Journal of Machine Learning Research, vol. 15, pp. 949-980, 2014.

[42] T. Salimans et al., "Evolution strategies as a scalable alternative to reinforcement learning," arXiv:1703.03864, 2017.

[43] M. Kumar et al., "Genetic algorithm: Review and application," SSRN, 2010.

[44] K. O. Stanley and R. Miikkulainen, "Evolving neural networks through augmenting topologies," Evolutionary Computation, vol. 10, no. 2, pp. 99-127, 2002.

[45] E. Real et al., "Large-scale evolution of image classifiers," ICML, pp. 2902-2911, 2017.

[46] B. Zoph and Q. V. Le, "Neural architecture search with reinforcement learning," ICLR, 2017.

[47] G. A. Vargas-Hakim et al., "A review on convolutional neural network encodings for neuroevolution," IEEE TEVC, vol. 26, no. 1, pp. 12-27, 2022.

[48] A. Baldominos et al., "Evolutionary convolutional neural networks: An application to handwriting recognition," Neurocomputing, vol. 283, pp. 38-52, 2018.

[49] P. Li et al., "Bridging evolutionary algorithms and reinforcement learning: A comprehensive survey," arXiv:2401.11963, 2024.

[50] S. Khadka and K. Tumer, "Evolution-guided policy gradient in reinforcement learning," NeurIPS, 2018.

[51] T. Gangwani and J. Peng, "Policy optimization by genetic distillation," ICLR, 2018.

[52] J. Singla et al., "SAPG: Split and aggregate policy gradients," ICML, pp. 45759-45772, 2024.

[53] D. Yarats et al., "Don't change the algorithm, change the data: Exploratory data for offline reinforcement learning," arXiv:2201.13425, 2022.

[54] A. Kumar et al., "Offline Q-learning on diverse multi-task data both scales and generalizes," ICLR, 2023.

[55] K. Wang et al., "Bootstrapped transformer for offline reinforcement learning," NeurIPS, pp. 34748-34761, 2022.

[56] C. Lu et al., "Synthetic experience replay," NeurIPS, pp. 46323-46344, 2023.

[57] R. Wang et al., "Prioritized generative replay," arXiv:2410.18082, 2024.

[58] K. Ota et al., "Can increasing input dimensionality improve deep reinforcement learning?" ICML, pp. 7424-7433, 2020.

[59] A. Bhatt et al., "CrossQ: Batch normalization in deep reinforcement learning for greater sample efficiency and simplicity," ICLR, 2024.

[60] H. Lee et al., "SimBa: Simplicity bias for scaling up parameters in deep reinforcement learning," arXiv:2410.09754, 2024.

[61] M. Nauman et al., "Bigger, regularized, optimistic: Scaling for compute and sample efficient continuous control," NeurIPS, 2024.

[62] M. Schwarzer et al., "Bigger, better, faster: Human-level Atari with human-level efficiency," ICML, pp. 30365-30380, 2023.

[63] S. Reed et al., "A generalist agent," Transactions on Machine Learning Research, 2024.

[64] E. Parisotto et al., "Stabilizing transformers for reinforcement learning," ICML, pp. 7487-7498, 2020.

[65] J. T. Springenberg et al., "Offline actor-critic reinforcement learning scales to large models," ICML, pp. 46323-46350, 2024.

[66] Y. Wang et al., "Scaling value iteration networks to 5000 layers for extreme long-term planning," arXiv:2406.08404, 2024.

[67] K. Wang et al., "1000 layer networks for self-supervised RL: Scaling depth can enable new goal-reaching capabilities," arXiv:2503.14858, 2025.

[68] I. Osband et al., "Deep exploration via bootstrapped DQN," NeurIPS, 2016.

[69] X. Chen et al., "Randomized ensembled double Q-learning: Learning fast without a model," ICLR, 2021.

[70] T. Hiraoka et al., "Dropout Q-functions for doubly efficient reinforcement learning," ICLR, 2022.

[71] Q. Lan et al., "Maxmin Q-learning: Controlling the estimation bias of Q-learning," ICLR, 2020.

[72] A. Kuznetsov et al., "Controlling overestimation bias with truncated mixture of continuous distributional quantile critics," ICML, pp. 5556-5566, 2020.

[73] J. Obando-Ceron et al., "Mixtures of experts unlock parameter scaling for deep RL," ICML, pp. 38520-38540, 2024.

[74] P. Li et al., "Bridging evolutionary algorithms and reinforcement learning: A comprehensive survey," arXiv:2401.11963, 2024.

[75] S. Khadka et al., "Collaborative evolutionary reinforcement learning," ICML, 2019.

[76] J. Hao et al., "ERL-Re2: Efficient evolutionary reinforcement learning with shared state representation and individual policy representation," ICLR, 2023.

[77] P. Li et al., "EvoRainbow: Combining improvements in evolutionary reinforcement learning for policy search," ICML, 2024.

[78] D. Kalashnikov et al., "Scalable deep reinforcement learning for vision-based robotic manipulation," CoRL, pp. 651-673, 2018.

[79] E. Nikishin et al., "The primacy bias in deep reinforcement learning," ICML, pp. 16828-16847, 2022.

[80] P. D'Oro et al., "Sample-efficient reinforcement learning by breaking the replay ratio barrier," ICLR, 2023.

[81] H. Lee et al., "PLASTIC: Improving input and label plasticity for sample efficient reinforcement learning," NeurIPS, pp. 62270-62295, 2023.

[82] Q. Li et al., "Efficient deep reinforcement learning requires regulating overfitting," ICLR, 2023.

[83] C. A. Voelcker et al., "MAD-TD: Model-augmented data stabilizes high update ratio RL," arXiv:2410.08896, 2024.

[84] Y. Yang et al., "Sample-efficient multiagent reinforcement learning with reset replay," ICML, 2024.

[85] T. Lahire et al., "Large batch experience replay," ICML, pp. 11790-11813, 2022.

[86] A. Nikulin et al., "Q-ensemble for offline RL: Don't scale the ensemble, scale the batch size," arXiv:2211.11092, 2022.

[87] D. Yarats et al., "Improving sample efficiency in model-free reinforcement learning from images," AAAI, pp. 10674-10681, 2021.

[88] M. Laskin et al., "CURL: Contrastive unsupervised representations for reinforcement learning," ICML, pp. 5639-5650, 2020.

[89] M. Jaderberg et al., "Reinforcement learning with unsupervised auxiliary tasks," ICLR, 2017.

[90] M. Schwarzer et al., "Data-efficient reinforcement learning with self-predictive representations," ICLR, 2021.

[91] M. Hoffman et al., "Acme: A research framework for distributed reinforcement learning," arXiv:2006.00979, 2020.

[92] J. Weng et al., "Tianshou: A highly modularized deep reinforcement learning library," JMLR, vol. 23, no. 267, pp. 1-6, 2022.

[93] Baidu, "PARL: A flexible, parallel and efficient reinforcement learning framework," 2022.

[94] J. Zhi et al., "Fiber: A platform for efficient development and distributed training for reinforcement learning and population-based methods," arXiv:2003.11164, 2020.

[95] A. Bou et al., "TorchRL: A data-driven decision-making library for PyTorch," 2023.

[96] Y. Mei et al., "SpeedyZero: Mastering Atari with limited data and time," ICLR, 2023.

[97] Z. Hu et al., "Asynchronous curriculum experience replay," IEEE TVT, vol. 72, no. 11, pp. 13985-14001, 2023.

[98] X. H. Liu et al., "Regret minimization experience replay in off-policy reinforcement learning," NeurIPS, pp. 17604-17615, 2021.

[99] C. Schwarke et al., "Curiosity-driven learning for joint locomotion and manipulation tasks," CoRL, 2023.

[100] S. Wu et al., "Quality-similar diversity via population based reinforcement learning," ICLR, 2022.

[101] E. Conti et al., "Improving exploration in evolution strategies for deep reinforcement learning via a population of novelty-seeking agents," NeurIPS, 2018.

[102] Y. Cao et al., "Survey on large language model-enhanced reinforcement learning: Concept, taxonomy, and methods," arXiv:2404.00282, 2024.

---
