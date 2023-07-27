"""
Pursuit-Evasion-O

##################################################
Task environment definition:

    1. Agent - Environment interaction model:
        - two-swarm turn taking: no.
        - (within swarm) agent-agent turn taking: no.

    2. Action execution model (collision mechanism):
        - agent-obstacle: bounce back and alive.
        - agent-agent (within swarm homogeneous): reach and alive.
        - pursuer-evader:
            - evader: reach and disappear.
            - pursuer: reach and alive.

        - (Remark: agent-agent, pursuer-evader switch position is also a kind of pursuer-evader collision
            since it is not possible in 2-D world without collisions.)

    3. Capture definition:
        - occupy based capture (and position switch based capture)
##################################################

Author: Lijun Sun.
"""

import numpy as np

from .matrix_world_two_swarm_concurrent_base import MatrixWorldTwoSwarmConcurrentBase


class MatrixWorld(MatrixWorldTwoSwarmConcurrentBase):

    def __init__(self,
                 world_rows=20, world_columns=20,
                 n_evaders=3, n_pursuers=12,
                 fov_scope=11,
                 max_env_cycles=500,
                 diagonal_move=False,
                 obstacle_density=0,
                 save_path="data/frames"):
        """
        :param world_rows: int, corresponds to the 1st axis.
        :param world_columns: int, corresponds to the 2nd axis.
        :param n_evaders: int, >= 0.
        :param n_pursuers: int, >= 0.
        :param fov_scope: int, >=1, an odd integer.
            The scope of the field of view of agents.
            The agent locates in the center of its own local field of view.
        :param obstacle_density: float.
        :param save_path: string.
        """

        super().__init__(world_rows=world_rows, world_columns=world_columns,
                         n_evaders=n_evaders, n_pursuers=n_pursuers,
                         fov_scope=fov_scope,
                         max_env_cycles=max_env_cycles,
                         diagonal_move=diagonal_move,
                         obstacle_density=obstacle_density,
                         save_path=save_path)

        # Reward parameters.

        # Reward pursuer.

        self.reward_pursuer_capture_evader = 10
        self.reward_pursuer_neighbor_evader = 0.1
        self.reward_pursuer_collide_with_obstacle = -12
        self.reward_pursuer_collide_with_pursuer = -12
        self.reward_pursuer_non_terminate = -0.05

        # Reward evader.

        self.reward_evader_being_captured = -10
        self.reward_evader_being_neighbored = -0.1
        self.reward_evader_collide_with_obstacle = -12
        self.reward_evader_collide_with_evader = -12
        self.reward_evader_non_terminate = 0.05

    def delete_captured_evaders(self):

        # Delete.

        self.evaders_last_active_index = self.evaders_active_index.copy()

        self.evaders_active_index = self.evaders_active_index[~self.evader_capture_status]

        all_positions_to_delete = self.evaders[self.evader_capture_status]

        all_positions_to_delete_in_padded = all_positions_to_delete + self.fov_offsets_in_padded

        # All evaders in this position are all captured and should be deleted.

        self.padded_env_matrix[all_positions_to_delete_in_padded[:, 0],
                               all_positions_to_delete_in_padded[:, 1], 0] = 0

        self.evaders_deleted_list += self.evaders[self.evader_capture_status].tolist()

        self.evaders = self.evaders[~self.evader_capture_status]

        # Update.

        self.n_evaders = len(self.evaders)

    def step(self, actions_pursuer=None, actions_evader=None):

        ##################################################
        # Check game status.

        if self.game_done:

            return_of_last_function = self.last()

            return return_of_last_function

        ##################################################
        # Reset

        self.reset_env_status()

        ##################################################
        # Pre-processing input.

        actions_pursuer = np.zeros(self.n_pursuers) if actions_pursuer is None else actions_pursuer

        actions_evader = np.zeros(self.n_evaders) if actions_evader is None else actions_evader

        ##################################################
        # Collision mechanism (action execution model).

        desired_positions_evader, desired_positions_pursuer = self.collision_mechanism(actions_evader,
                                                                                       actions_pursuer)

        ##################################################
        # Swarm move.

        # Evaders.

        to_positions_evader = np.array(desired_positions_evader)

        for idx_evader, to_position in enumerate(to_positions_evader):

            self.update_a_evader(idx_evader, to_position)

        # Pursuers.

        to_positions_pursuer = np.array(desired_positions_pursuer)

        for idx_pursuer, to_position in enumerate(to_positions_pursuer):

            self.update_a_pursuer(idx_pursuer, to_position)

        ##################################################
        # Update game status and performance monitors.

        self.update_game_status()

        # Evaders.

        self.n_collisions_evaders_with_obstacles = sum(self.evader_collision_status_with_obstacles)
        self.n_collisions_evaders_with_evaders = sum(self.evader_collision_status_with_evaders)
        self.n_collisions_evaders_with_pursuers = sum(self.evader_collision_status_with_pursuers)

        # Pursuers.

        self.n_collisions_pursuers_with_obstacles = sum(self.pursuer_collision_status_with_obstacles)
        self.n_collisions_pursuers_with_pursuers = sum(self.pursuer_collision_status_with_pursuers)
        self.n_collisions_pursuers_with_evaders = sum(self.pursuer_collision_status_with_evaders)

        ##################################################
        # Reward.

        self.reward_evaders, self.reward_pursuers = self.get_reward()

        ##################################################
        # Delete captured evaders.

        self.delete_captured_evaders()

        ##################################################
        # Return.

        return_of_last_function = self.last()

        return return_of_last_function

    def collision_mechanism(self, actions_evader, actions_pursuer):
        """
        Action execution model.
        """

        ##################################################
        # 1. Collision: agent-obstacle.

        from_positions_evader, desired_positions_evader = \
            self.collision_mechanism_agent_obstacle_bounce_back_alive(actions_evader, is_evader=True)

        from_positions_pursuer, desired_positions_pursuer = \
            self.collision_mechanism_agent_obstacle_bounce_back_alive(actions_pursuer, is_evader=False)

        ##################################################
        # 2. Collision: agent-agent.

        from_positions_evader, desired_positions_evader = \
            self.collision_mechanism_agent_agent_reach(from_positions_evader, desired_positions_evader, is_evader=True)

        from_positions_pursuer, desired_positions_pursuer = \
            self.collision_mechanism_agent_agent_reach(from_positions_pursuer, desired_positions_pursuer,
                                                       is_evader=False)

        ##################################################
        # 3. Collision: pursuer-evader.

        desired_positions_evader, desired_positions_pursuer = \
            self.collision_mechanism_evader_reach_pursuer_reach(from_positions_evader,
                                                                desired_positions_evader,
                                                                from_positions_pursuer,
                                                                desired_positions_pursuer)

        return desired_positions_evader, desired_positions_pursuer

    def update_game_status(self):

        self.evader_capture_status = self.evader_collision_status_with_pursuers

        self.pursuer_capture_status = self.pursuer_collision_status_with_evaders

        self.capture_rate = \
            (self.n_evaders_initial - self.n_evaders + sum(self.evader_capture_status)) / self.n_evaders_initial

        self.evader_survival_rate = (self.n_evaders - sum(self.evader_capture_status)) / self.n_evaders_initial

        self.game_done = sum(self.evader_capture_status) == len(self.evader_capture_status)

        self.evaders_done = self.evader_capture_status.copy()

        self.pursuers_done = np.logical_or(self.game_done, self.pursuers_done)

        self.env_step_counter += 1

    def get_reward(self):

        reward_evader_being_captured, reward_pursuer_capture = self.get_reward_capture_evader()
        reward_evader_being_neighbored, reward_pursuer_neighbor_evader = self.get_reward_neighbor_evader()
        reward_evader_collide, reward_pursuer_collide = self.get_reward_collide()

        reward_evaders = \
            reward_evader_being_captured + \
            reward_evader_being_neighbored + \
            reward_evader_collide + \
            self.reward_evader_non_terminate

        reward_pursuers = \
            reward_pursuer_capture + \
            reward_pursuer_neighbor_evader + \
            reward_pursuer_collide + \
            self.reward_pursuer_non_terminate

        return reward_evaders, reward_pursuers

    def get_reward_capture_evader(self):

        reward_evader = np.zeros(self.n_evaders)

        reward_pursuer = np.zeros(self.n_pursuers)

        reward_evader[self.evader_capture_status] = self.reward_evader_being_captured

        reward_pursuer[self.pursuer_capture_status] = self.reward_pursuer_capture_evader

        return reward_evader, reward_pursuer

    def get_reward_neighbor_evader(self):

        reward_evader = np.zeros(self.n_evaders)
        reward_pursuer = np.zeros(self.n_pursuers)

        for idx_evader in range(self.n_evaders):

            position = self.get_a_evader(idx_evader)

            axial_neighbors = self.axial_neighbors_mask + position + self.fov_offsets_in_padded

            n_axial_neighbor_pursuers = self.padded_env_matrix[axial_neighbors[:, 0], axial_neighbors[:, 1], 1].sum()

            if n_axial_neighbor_pursuers > 0:

                reward_evader[idx_evader] = self.reward_evader_being_neighbored

        for idx_pursuer in range(self.n_pursuers):

            position = self.get_a_pursuer(idx_pursuer)

            axial_neighbors = self.axial_neighbors_mask + position + self.fov_offsets_in_padded

            n_axial_neighbor_evaders = self.padded_env_matrix[axial_neighbors[:, 0], axial_neighbors[:, 1], 0].sum()

            if n_axial_neighbor_evaders > 0:

                reward_pursuer[idx_pursuer] = self.reward_pursuer_neighbor_evader

        return reward_evader, reward_pursuer

    def get_reward_collide(self):

        reward_evader = \
            self.reward_evader_collide_with_obstacle * self.evader_collision_status_with_obstacles + \
            self.reward_evader_collide_with_evader * self.evader_collision_status_with_evaders

        reward_pursuer = \
            self.reward_pursuer_collide_with_obstacle * self.pursuer_collision_status_with_obstacles + \
            self.reward_pursuer_collide_with_pursuer * self.pursuer_collision_status_with_pursuers

        return reward_evader, reward_pursuer

    pass

