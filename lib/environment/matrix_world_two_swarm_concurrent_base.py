"""
Base class for the two-swarm concurrent game in the matrix world.
- Two swarm, i.e., all agents, concurrently observe, make decision, and execute action.

##################################################
Environment performance status:

    - evader
        - evader capture status.
        - evader collision with obstacle.
        - evader collision with evader.
        - evader collision with pursuer.
    - pursuer
        - pursuer collision with obstacle.
        - pursuer collision with pursuer.
        - pursuer collision with evader.
    - capture rate.
##################################################

Author: Lijun Sun.
"""

import copy
import numpy as np

from .matrix_world_base import MatrixWorldBase


class MatrixWorldTwoSwarmConcurrentBase(MatrixWorldBase):

    def __init__(self,
                 world_rows=20, world_columns=20,
                 n_evaders=3, n_pursuers=12,
                 fov_scope=11,
                 max_env_cycles=500,
                 diagonal_move=False,
                 obstacle_density=0,
                 save_path="data/frames"):

        super().__init__(world_rows=world_rows, world_columns=world_columns,
                         n_evaders=n_evaders, n_pursuers=n_pursuers,
                         fov_scope=fov_scope,
                         max_env_cycles=max_env_cycles,
                         diagonal_move=diagonal_move,
                         obstacle_density=obstacle_density,
                         save_path=save_path)

        # Env status: share parameters between functions.

        self.reward_pursuers = np.zeros(self.n_pursuers)

        self.reward_evaders = np.zeros(self.n_evaders)

        self.pursuer_capture_status = np.array([False] * self.n_pursuers)
        self.pursuer_collision_status = np.array([False] * self.n_pursuers)
        self.pursuer_collision_status_with_obstacles = np.array([False] * self.n_pursuers)
        self.pursuer_collision_status_with_pursuers = np.array([False] * self.n_pursuers)
        self.pursuer_collision_status_with_evaders = np.array([False] * self.n_pursuers)

        self.evader_capture_status = np.array([False] * self.n_evaders)
        self.evader_collision_status = np.array([False] * self.n_evaders)
        self.evader_collision_status_with_obstacles = np.array([False] * self.n_evaders)
        self.evader_collision_status_with_pursuers = np.array([False] * self.n_evaders)
        self.evader_collision_status_with_evaders = np.array([False] * self.n_evaders)

        self.capture_rate = 0
        self.evader_survival_rate = 0
        self.pursuer_survival_rate = 0

        self.n_collisions_pursuers_with_obstacles = 0
        self.n_collisions_pursuers_with_pursuers = 0
        self.n_collisions_pursuers_with_evaders = 0

        self.n_collisions_evaders_with_obstacles = 0
        self.n_collisions_evaders_with_pursuers = 0
        self.n_collisions_evaders_with_evaders = 0

        pass

    def reset_env_status(self):

        # Share parameters between functions.

        self.reward_pursuers = np.zeros(self.n_pursuers)

        self.reward_evaders = np.zeros(self.n_evaders)

        self.pursuer_capture_status = np.array([False] * self.n_pursuers)
        self.pursuer_collision_status = np.array([False] * self.n_pursuers)
        self.pursuer_collision_status_with_obstacles = np.array([False] * self.n_pursuers)
        self.pursuer_collision_status_with_pursuers = np.array([False] * self.n_pursuers)
        self.pursuer_collision_status_with_evaders = np.array([False] * self.n_pursuers)

        self.evader_capture_status = np.array([False] * self.n_evaders)
        self.evader_collision_status = np.array([False] * self.n_evaders)
        self.evader_collision_status_with_obstacles = np.array([False] * self.n_evaders)
        self.evader_collision_status_with_pursuers = np.array([False] * self.n_evaders)
        self.evader_collision_status_with_evaders = np.array([False] * self.n_evaders)

        self.capture_rate = 0
        self.evader_survival_rate = 0
        self.pursuer_survival_rate = 0

        self.n_collisions_pursuers_with_obstacles = 0
        self.n_collisions_pursuers_with_pursuers = 0
        self.n_collisions_pursuers_with_evaders = 0

        self.n_collisions_evaders_with_obstacles = 0
        self.n_collisions_evaders_with_pursuers = 0
        self.n_collisions_evaders_with_evaders = 0

    def last(self):
        """
        :return:
            - observation: (n_agent, fov_scope, fov_scope, 3)
        """

        ##################################################
        # Observe.

        observation_swarm_evader = self.perceive_swarm_local_matrix(is_evader=True)
        observation_swarm_pursuer = self.perceive_swarm_local_matrix(is_evader=False)

        observation_swarm = dict(evader=observation_swarm_evader,
                                 pursuer=observation_swarm_pursuer)

        ##################################################
        # Reward.

        reward_swarm = dict(evader=self.reward_evaders,
                            pursuer=self.reward_pursuers)

        ##################################################
        # Done.

        done_swarm = dict(game_done=self.game_done,
                          evader=self.evaders_done,
                          pursuer=self.pursuers_done)

        ##################################################
        # Info.

        info = dict(  # game status.
                    capture_rate=self.capture_rate,
                    evader_survival_rate=self.evader_survival_rate,
                    pursuer_survival_rate=self.pursuer_survival_rate,
                    evader_capture_status=self.evader_capture_status.copy(),
                    # collision status.
                    n_collisions_evaders_with_obstacles=self.n_collisions_evaders_with_obstacles,
                    n_collisions_evaders_with_evaders=self.n_collisions_evaders_with_evaders,
                    n_collisions_evaders_with_pursuers=self.n_collisions_evaders_with_pursuers,
                    n_collisions_pursuers_with_obstacles=self.n_collisions_pursuers_with_obstacles,
                    n_collisions_pursuers_with_pursuers=self.n_collisions_pursuers_with_pursuers,
                    n_collisions_pursuers_with_evaders=self.n_collisions_pursuers_with_evaders,
                    # alive status.
                    evaders_last_active_index=self.evaders_last_active_index.copy(),
                    evaders_active_index=self.evaders_active_index.copy(),
                    pursuers_last_active_index=self.pursuers_last_active_index.copy(),
                    pursuers_active_index=self.pursuers_active_index.copy())

        return copy.deepcopy(observation_swarm), copy.deepcopy(reward_swarm), done_swarm, copy.deepcopy(info)

    def step(self, swarm_actions_pursuer=None, swarm_actions_evader=None):
        """
        Evader swarm and pursuer swarm move simultaneously.
        """
        raise NotImplementedError

    def get_reward(self):
        raise NotImplementedError

    def update_game_status(self):
        raise NotImplementedError

    ##################################################
    # Workflow in implementing a collision mechanism:
    # 1. Detect two types of conflicts:
    #    (1) vertex conflict, (2) edge conflict (not applicable to agent-obstacle collision).
    # 2. Resolve conflict based on the safety definition.
    # 3. Record conflict, i.e., whose fault it is.

    ##################################################
    # 1. Collision mechanism: agent-obstacle.

    def collision_mechanism_agent_obstacle_reach(self, actions, is_evader=False):
        """
        1. Collision: agent-obstacle.
           Collision outcome: reach and disappear.
        """

        from_positions = self.get_all_evaders() if is_evader else self.get_all_pursuers()

        desired_positions = []

        for idx, action in enumerate(actions.tolist()):

            from_position = from_positions[idx]

            desired_position = self.move_to(from_position, action)

            if self.is_collide_with_obstacle(desired_position):

                if is_evader:

                    self.evader_collision_status_with_obstacles[idx] = True

                else:

                    self.pursuer_collision_status_with_obstacles[idx] = True

            desired_positions.append(list(desired_position))

        return from_positions.copy(), desired_positions.copy()

    def collision_mechanism_agent_obstacle_bounce_back_alive(self, actions, is_evader=False):
        """
        1. Collision: agent-obstacle.
           Collision outcome: bounce back.
        """

        from_positions = self.get_all_evaders() if is_evader else self.get_all_pursuers()

        desired_positions = []

        for idx, action in enumerate(actions.tolist()):

            from_position = from_positions[idx]

            desired_position = self.move_to(from_position, action)

            if self.is_collide_with_obstacle(desired_position):

                # Collision outcome: bounce back.

                desired_position = from_position

                if is_evader:

                    self.evader_collision_status_with_obstacles[idx] = True

                else:

                    self.pursuer_collision_status_with_obstacles[idx] = True

            desired_positions.append(list(desired_position))

        return from_positions.copy(), desired_positions.copy()

    ##################################################
    # 2. Collision mechanism: agent-agent.

    def collision_mechanism_agent_agent_reach(self, from_positions, desired_positions, is_evader=False):
        """
        2. Collision: agent-agent (within swarm homogeneous).
           Collision outcome: reach, disappear or alive is post-processing.
        """

        for idx_1, (from_position_1, desired_position_1) in enumerate(zip(from_positions.tolist(), desired_positions)):

            for idx_2, (from_position_2, desired_position_2) in enumerate(zip(from_positions.tolist(),
                                                                              desired_positions)):

                if idx_1 == idx_2:
                    continue

                switch_based_collision = (desired_position_1 == from_position_2) and \
                                         (from_position_1 == desired_position_2)

                occupy_based_collision = (desired_position_1 == desired_position_2)

                if switch_based_collision or occupy_based_collision:

                    if is_evader:

                        self.evader_collision_status_with_evaders[idx_1] = True

                    else:

                        self.pursuer_collision_status_with_pursuers[idx_1] = True

        return from_positions.copy(), desired_positions.copy()

    def collision_mechanism_agent_agent_bounce_back_alive(self, from_positions, desired_positions, is_evader=False):
        """
        2. Collision: agent-agent (within swarm homogeneous).
           Collision outcome: bounce back and alive.
        """

        n_agents = self.n_evaders if is_evader else self.n_pursuers

        index_bounce_back_or_not = [False] * n_agents

        for idx_1, (from_position_1, desired_position_1) in enumerate(zip(from_positions.tolist(), desired_positions)):

            for idx_2, (from_position_2, desired_position_2) in enumerate(zip(from_positions.tolist(),
                                                                              desired_positions)):

                if idx_1 == idx_2:
                    continue

                switch_based_collision = (desired_position_1 == from_position_2) and \
                                         (from_position_1 == desired_position_2)

                occupy_based_collision = (desired_position_1 == desired_position_2)

                if switch_based_collision or occupy_based_collision:

                    index_bounce_back_or_not[idx_1] = True

                    if is_evader:

                        self.evader_collision_status_with_evaders[idx_1] = True

                    else:

                        self.pursuer_collision_status_with_pursuers[idx_1] = True

        desired_positions = np.array(desired_positions)
        desired_positions[index_bounce_back_or_not] = from_positions[index_bounce_back_or_not]
        desired_positions = desired_positions.tolist()

        return from_positions.copy(), desired_positions.copy()

    ##################################################
    # 3. Collision mechanism: pursuer-evader.

    def collision_mechanism_evader_reach_pursuer_reach(
            self, from_positions_evader, desired_positions_evader, from_positions_pursuer, desired_positions_pursuer):
        """
        3. Collision: pursuer-evader.
           Collision outcome: disappear or alive is post-processed.
                - evader: reach.
                - pursuer: reach.
        """

        for idx_evader, (from_position_evader, desired_position_evader) in \
                enumerate(zip(from_positions_evader.tolist(), desired_positions_evader)):

            for idx_pursuer, (from_position_pursuer, desired_position_pursuer) in \
                    enumerate(zip(from_positions_pursuer.tolist(), desired_positions_pursuer)):

                switch_based_collision = (desired_position_evader == from_position_pursuer) and \
                                         (from_position_evader == desired_position_pursuer)

                occupy_based_collision = (desired_position_evader == desired_position_pursuer)

                if switch_based_collision or occupy_based_collision:

                    self.evader_collision_status_with_pursuers[idx_evader] = True

                    self.pursuer_collision_status_with_evaders[idx_pursuer] = True

        return desired_positions_evader.copy(), desired_positions_pursuer.copy()

    def collision_mechanism_evader_bounce_back_alive_pursuer_reach_alive(
            self, from_positions_evader, desired_positions_evader, from_positions_pursuer, desired_positions_pursuer):
        """
        3. Collision: pursuer-evader.
           Collision outcome:
                - evader: bounce back and alive. lower priority.
                - pursuer: reach and alive. higher priority.
        """

        # (1) If evader keeps still, the pursuer will fail to reach that position and bounce back,
        # since no vertex conflict is allowed in this case,
        # and it is the fault of the pursuer.

        still_evader_idx = []

        for idx_evader, (from_position_evader, desired_position_evader) in \
                enumerate(zip(from_positions_evader.tolist(), desired_positions_evader)):

            if desired_position_evader == from_position_evader:

                still_evader_idx.append(desired_position_evader)

        for idx_pursuer, position in enumerate(desired_positions_pursuer):

            if position in still_evader_idx:

                # Pursuer bounce back since it cannot reach an occupied position.

                desired_positions_pursuer[idx_pursuer] = from_positions_pursuer[idx_pursuer]

                self.pursuer_collision_status_with_evaders[idx_pursuer] = True

        desired_positions_pursuer = np.array(desired_positions_pursuer)

        # (2) If there is an edge conflict between an evader and a pursuer, both will bounce back
        # since no vertex and edge conflicts are allowed in this case,
        # and it is the fault of both.

        index_bounce_back_or_not_evader = [False] * self.n_evaders
        index_bounce_back_or_not_pursuer = [False] * self.n_pursuers

        for idx_evader, (from_position_evader, desired_position_evader) in \
                enumerate(zip(from_positions_evader.tolist(), desired_positions_evader)):

            #  Only consider moving evader here.

            if from_position_evader in still_evader_idx:
                continue

            for idx_pursuer, (from_position_pursuer, desired_position_pursuer) in \
                    enumerate(zip(from_positions_pursuer.tolist(), desired_positions_pursuer)):

                switch_based_collision = (desired_position_evader == from_position_pursuer) and \
                                         (from_position_evader == desired_position_pursuer)

                if switch_based_collision:

                    index_bounce_back_or_not_evader[idx_evader] = True
                    index_bounce_back_or_not_pursuer[idx_pursuer] = True

                    self.evader_collision_status_with_pursuers[idx_evader] = True
                    self.pursuer_collision_status_with_evaders[idx_pursuer] = True

        desired_positions_evader = np.array(desired_positions_evader)
        desired_positions_evader[index_bounce_back_or_not_evader] = from_positions_evader[index_bounce_back_or_not_evader]
        desired_positions_evader = desired_positions_evader.tolist()

        desired_positions_pursuer = np.array(desired_positions_pursuer)
        desired_positions_pursuer[index_bounce_back_or_not_evader] = from_positions_pursuer[index_bounce_back_or_not_pursuer]
        desired_positions_pursuer = desired_positions_pursuer.tolist()

        # (3) Otherwise, if there is only a vertex conflict between a moving evader and a moving pursuer,
        # the evader will bounce back due to lower priority and pursuer will reach due to higher priority,
        # and it is the fault of the evader.

        for idx_evader, (from_position_evader, desired_position_evader) in \
                enumerate(zip(from_positions_evader.tolist(), desired_positions_evader)):

            #  Only consider moving evader here.

            if from_position_evader in still_evader_idx:
                continue

            if desired_position_evader in desired_positions_pursuer.tolist():

                # Evader fail to take the desired position over pursuer.

                desired_positions_evader[idx_evader] = from_position_evader

                self.evader_collision_status_with_pursuers[idx_evader] = True

        return desired_positions_evader.copy(), desired_positions_pursuer.copy()

    pass

