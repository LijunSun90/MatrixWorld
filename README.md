
Official code for the paper 
"MatrixWorld: A pursuit-evasion platform for safe multi-agent coordination and autocurricula", 
which is preprinted in [Arxiv](https://arxiv.org/abs/2307.14854) and under review.


**More documents will be updated continuously.**


## Description

MatrixWorld is 
- a safety constrained pursuit-evasion platform for safe multi-agent coordination,
- a lightweight co-evolution environment for autocurricula research.

In this work, 
- the safety is defined in terms of multi-agent collision avoidance.
It covers diverse safety definitions in the real-world applications.
- 9 pursuit-evasion game variants are defined for example scenarios
like real-world drone and vehicle swarm,
multi-agent path finding (MAPF), 
popular pursuit-evasion setups, 
and classic cops-and-robbers problem.

It can be used for the research of
- safe multi-agent environment implementation,
- safe multi-agent reinforcement learnng (MARL),
- safe multi-agent coordination,
- co-evolution, autocurricula, self-play, arm races, or adversarial learning.


## Task definition

- Nine pursuit-evasion variants are defined for example **scenarios** like 
  **(1)** real-world drone and vehicle swarm, 
  **(2)** multi-agent path finding (MAPF), 
  **(3)** popular pursuit-evasion setups, and 
  **(4)** classic cops-androbbers problem.
- **More** pursuit-evasion variants (other tasks) **can be designed** based on different practical meanings of safety.

![Alt Text](https://github.com/LijunSun90/MatrixWorld/blob/main/docs/figures/task_definition.png)


### Environmental parameters

- **Origin** of the grid world: Top left corner.
- **Size** of the grid world: Tunable. Default value: 20 x 20.
- **Swarm size** of agents (pursuers and evaders): Tunable. Default value: n_evaders=3, n_pursuers=12.
- **Observation:** Tunable. Binary **matrix** of size **fov_scope x fov_scope x 3**, 
  which is a square centered at the agent with 3 channels: local_evader, local_pursuers, local_obstacles.
  Default value is: 11 x 11 x 3.
- **Action**: Vector of size **5**, where 0 ~ 4 represent keeping still, moving north, moving east, moving south, and moving west.

**Remark:** The codes provide utility functions for matrix-based and vector-based global and local observations.

## Safety-constrained multi-agent action execution model

The proposed safety-constrained multi-agent action execution model is **general for the software implementation of safe multi-agent environments**.

It consists **two parts: (1)** multi-agent-environment interaction model; 
**(2)** safety-constrained collision resolution mechanism for the simultaneous action execution of multiple agents.

### (1) Multi-agent-environment interaction model

Multi-agent-environment interaction model in adversarial multi-agent settings, e.g., pursuit-evasion games.

![Alt Text](https://github.com/LijunSun90/MatrixWorld/blob/main/docs/figures/multiagent_environment_interaction_models.png)

### (2) Safety-constrained collision resolution mechanism

The **collision resolution mechanism is defined for the simultaneous action execution of agents**, 
which consists of 3 collisions types and 3 collision outcomes for each type,
based on the safety definitions in real-world applications and literature conventions.

**Remark:** 
The collision resolution mechanism also determines which agent should be responsible, 
which is useful for the correct learning of algorithms.

![Alt Text](https://github.com/LijunSun90/MatrixWorld/blob/main/docs/figures/conflict_type.png)
![Alt Text](https://github.com/LijunSun90/MatrixWorld/blob/main/docs/figures/collision_mechanism.png)

## Lightweight co-evolution platform

- MatrixWorld is a lightweight co-evolution platform to test autocurricula research ideas.
- Our experiments achieve the autocurricula between pursuers and evaders by adversarial learning. 
- Our experiments show that 
  the passive (evasive) policy learning benefits more from co-evolution 
  than active (pursuing) policy learning in an asymmetric adversarial game.

Figure: **(left)** evasive behavior trained by normal reinforcement learning;
**(middle)** evasive behavior trained by adversarial learning;
**(right)** arms race in the learning process of pursuers and evaders.

![Alt Text](https://github.com/LijunSun90/MatrixWorld/blob/main/data/o_compare/video_evasion_trained_by_random_vs_adversarial.gif)
![Alt Text](https://github.com/LijunSun90/MatrixWorld/blob/main/data/o_compare/video_evasion_trained_by_adversarial_vs_adversarial.gif)
![Alt Text](https://github.com/LijunSun90/MatrixWorld/blob/main/docs/figures/capture_rate_generalist_vs_generalist.png)


## Paper citation

Cite the following paper if you use this environment, code, or found it useful.

    @article{sun2023matrixworld,
      title={MatrixWorld: A pursuit-evasion platform for safe multi-agent coordination and autocurricula},
      author={Sun, Lijun and Chang, Yu-Cheng and Lyu, Chao and Lin, Chin-Teng and Shi, Yuhui},
      journal={arXiv preprint arXiv:2307.14854},
      year={2023}
    }

