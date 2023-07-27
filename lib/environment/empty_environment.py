import copy
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
        super().__init__(world_rows=world_rows, world_columns=world_columns,
                         n_evaders=n_evaders, n_pursuers=n_pursuers,
                         fov_scope=fov_scope,
                         max_env_cycles=max_env_cycles,
                         diagonal_move=diagonal_move,
                         obstacle_density=obstacle_density,
                         save_path=save_path)

    def reset_env_status(self):
        pass

    def last(self):
        """
        :return:
            - observation: (n_agent, fov_scope, fov_scope, 3)
        """

        ##################################################
        # Observe. Empty.

        observation_swarm_evader = np.zeros((self.n_evaders, self.fov_scope, self.fov_scope, 3), dtype=int)
        observation_swarm_pursuer = np.zeros((self.n_pursuers, self.fov_scope, self.fov_scope, 3), dtype=int)

        observation_swarm = dict(evader=observation_swarm_evader,
                                 pursuer=observation_swarm_pursuer)

        ##################################################
        # Reward.

        reward_swarm = None

        ##################################################
        # Done.

        done_swarm = dict(game_done=self.game_done,
                          evader=self.evaders_done,
                          pursuer=self.pursuers_done)

        ##################################################
        # Info.

        info = None

        return copy.deepcopy(observation_swarm), copy.deepcopy(reward_swarm), done_swarm, copy.deepcopy(info)

    def step(self, actions_pursuer=None, actions_evader=None):
        """
        Evader swarm and pursuer swarm move simultaneously.
        """

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

        ##################################################
        # Reward.

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

        self.env_step_counter += 1

    def get_reward(self):
        pass

