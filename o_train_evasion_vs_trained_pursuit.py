import os
import time
import argparse
import copy

import random
import numpy as np

import torch

from lib.environment.pursuit_evasion_o import MatrixWorld as Environment
from lib.utils.experiment_logger import ExperimentLogger
from lib.utils.models import ModelMLP
from lib.utils.models import ModelPolicySafe
from lib.utils.models import ModelPolicy
from lib.utils.models import ModelActorCritic
from lib.actor_critic.trainer_agent import TrainerAgent


def parse_args():

    parser = argparse.ArgumentParser()

    ##################################################
    # Log parameters.

    data_log_folder = os.path.join("data", os.path.basename(__file__)[:-3])
    os.makedirs(data_log_folder, exist_ok=True)
    parser.add_argument("--data_log_folder", type=str, default=data_log_folder)

    parser.add_argument("--resume_model_evader", action="store_true", default=False)
    parser.add_argument("--resume_model_name_actor_evader", type=str,
                        default=os.path.join(data_log_folder, 'model_actor_evader.pth'))
    parser.add_argument("--resume_model_name_critic_evader", type=str,
                        default=os.path.join(data_log_folder, 'model_critic_evader.pth'))

    parser.add_argument("--resume_model_pursuer", action="store_true", default=True)
    parser.add_argument("--resume_model_name_actor_pursuer", type=str,
                        default=os.path.join(data_log_folder, 'model_actor_pursuer.pth'))
    parser.add_argument("--resume_model_name_critic_pursuer", type=str,
                        default=os.path.join(data_log_folder, 'model_critic_pursuer.pth'))

    ##################################################
    # Environment parameters.

    parser.add_argument("--world_size", type=int, default=40)
    parser.add_argument("--n_pursuers", type=int, default=8)
    parser.add_argument("--n_evaders", type=int, default=30)

    ##################################################
    # General experiment parameters.

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--render", type=bool, default=False)

    parser.add_argument("--n_epochs", type=int, default=5000,
                        help="Epoch is different from episode. "
                             "An epoch can collect experiences of more than one or less than one episodes.")

    parser.add_argument("--steps_per_epoch", type=int, default=500,
                        help="steps_per_epoch is different from max_episode_length.")

    parser.add_argument("--max_episode_length", type=int, default=500)

    if torch.cuda.device_count() > 1:
        device = torch.device('cuda:0')
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    parser.add_argument("--device", type=torch.device, default=device)

    parser.add_argument("--save_model_frequency", type=int, default=100)
    parser.add_argument("--model_name_frequency", type=int, default=500)
    parser.add_argument("--log_performance_frequency", type=int, default=1,
                        help="time duration between successive log printing.")

    ##################################################
    # Model parameters.

    parser.add_argument("--use_initialization", action='store_false', default=True,
                        help="True if not specified. Use it will converge faster without loss of final performance.")

    parser.add_argument("--with_layer_normalization", action='store_false', default=False,
                        help="True if not specified. Generally, with it, learning is more stable and better. "
                             "But not good for pursuit-evasion-o (inefficient learning, unstable).")

    parser.add_argument("--with_obstacle_avoidance_mask", action='store_false', default=False,
                        help="True if not specified. Perfect obstacle avoidance. "
                             "But not good for pursuit-evasion-o (less stable).")
    ##################################################
    # Algorithm parameters.

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lam", type=float, default=0.97,
                        help="General advantage estimation (GAE) parameter.")
    parser.add_argument("--lr_policy", type=float, default=3e-4)
    parser.add_argument("--lr_value", type=float, default=1e-3)
    parser.add_argument("--train_iterations_policy", type=int, default=1)
    parser.add_argument("--train_iterations_value", type=int, default=5)
    parser.add_argument("--entropy_coefficient", type=float, default=0, help="or try 0.01")
    parser.add_argument("--use_gradient_norm", action='store_true', default=False, help="Trick")
    parser.add_argument("--max_gradient_norm", type=float, default=10, help="Trick")

    return parser.parse_args()


class RunExperiment:

    def __init__(self, args):

        self.args = args

        self.logger = ExperimentLogger(self.args.data_log_folder)

        # Share parameters.

        # Value may change with time.

        self.n_evaders = self.args.n_evaders
        self.n_pursuers = self.args.n_pursuers

        # step_1_create_environment.

        self.env = None
        self.dim_observation = None
        self.dim_action = None

        # step_3_create_trainer_agent

        self.evader = None
        self.pursuer = None

        # step_4_experiment_over_epoch

        self.epoch_counter = 0

        self.episode_timestep_counter = 0

        self.dim_observation_actor = 0

        self.dim_observation_critic = 0

        # step_4_2_update_model

        self.loss_value_evader = 0
        self.loss_policy_evader = 0

        pass

    def run(self):

        self.step_1_create_environment()

        self.step_2_set_random_seed()

        self.step_3_create_trainer_agent()

        self.step_4_experiment_over_epoch()

    def step_1_create_environment(self):

        self.env = Environment(world_rows=self.args.world_size, world_columns=self.args.world_size,
                               n_evaders=self.args.n_evaders, n_pursuers=self.args.n_pursuers,
                               fov_scope=11,
                               max_env_cycles=self.args.max_episode_length,
                               save_path=os.path.join(self.args.data_log_folder, "frames"))

        self.dim_observation_actor = (self.env.fov_scope, self.env.fov_scope, 3)

        self.dim_observation_critic = self.env.fov_scope * self.env.fov_scope * 3

        self.dim_action = self.env.n_actions

    def step_2_set_random_seed(self):
        """
        - First, create an environment instance.
        - Second, set random seed, including that for the environment.
        - Third, do all the other things.
        """
        seed = self.args.seed + 10000
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.env.reset(seed=seed)

    def step_3_create_trainer_agent(self):

        # Use args parameter to initialize.

        self.evader = self.step_3_1_create_trainer_agent(self.args.n_evaders, is_evader=True)
        self.pursuer = self.step_3_1_create_trainer_agent(self.args.n_pursuers, is_evader=False)

    def step_3_1_create_trainer_agent(self, n_agents, is_evader=False):

        model_actor_critic = self.step_3_1_1_initialize_model_actor_critic(dim_input_actor=self.dim_observation_actor,
                                                                           dim_input_critic=self.dim_observation_critic,
                                                                           dim_output=self.dim_action,
                                                                           is_evader=is_evader)

        agents = TrainerAgent(model_actor_critic=model_actor_critic,
                              buffer_size=self.args.steps_per_epoch,
                              n_agents=n_agents,
                              dim_observation_actor=self.dim_observation_actor,
                              dim_observation_critic=self.dim_observation_critic,
                              configuration=self.args)

        return agents

    def step_3_1_1_initialize_model_actor_critic(self, dim_input_actor, dim_input_critic, dim_output, is_evader=False):

        # Policy.

        if self.args.with_obstacle_avoidance_mask:

            model_policy = ModelPolicySafe(dim_input=dim_input_actor,
                                           dim_output=dim_output,
                                           hidden_sizes=(400, 300),
                                           with_layer_normalization=self.args.with_layer_normalization,
                                           use_initialization=self.args.use_initialization,
                                           obstacle_channel=[2])

        else:

            model_policy = ModelPolicy(dim_input=dim_input_actor,
                                       dim_output=dim_output,
                                       hidden_sizes=(400, 300),
                                       with_layer_normalization=self.args.with_layer_normalization,
                                       use_initialization=self.args.use_initialization)

        # Value.

        model_value = ModelMLP(dim_input=dim_input_critic, dim_output=1, hidden_sizes=(400, 300),
                               with_layer_normalization=self.args.with_layer_normalization,
                               use_initialization=self.args.use_initialization)

        if is_evader and self.args.resume_model_evader:

            model_policy.load_state_dict(torch.load(self.args.resume_model_name_actor_evader,
                                                    map_location=self.args.device))

            model_value.load_state_dict(torch.load(self.args.resume_model_name_critic_evader,
                                                   map_location=self.args.device))

        elif (not is_evader) and self.args.resume_model_pursuer:

            model_policy.load_state_dict(torch.load(self.args.resume_model_name_actor_pursuer,
                                                    map_location=self.args.device))

            model_value.load_state_dict(torch.load(self.args.resume_model_name_critic_pursuer,
                                                   map_location=self.args.device))

        # Actor critic.

        model_actor_critic = ModelActorCritic(model_policy=model_policy,
                                              model_value=model_value).to(self.args.device)

        return model_actor_critic

    def step_4_experiment_over_epoch(self):

        for i_epoch in range(self.args.n_epochs):

            start_epoch_time = time.time()

            self.step_4_1_experiment_of_an_epoch()

            self.step_4_2_update_model()

            self.step_4_3_log_info_of_epoch(time.time() - start_epoch_time)

            # Update.

            self.epoch_counter += 1

            pass

    def step_4_1_experiment_of_an_epoch(self):

        observations, rewards, game_done, info = self.env.reset()

        for timestep in range(self.args.steps_per_epoch):

            if self.args.render:
                self.env.render()

            # 1. Agents make decisions.

            observations_actor_evader, observations_critic_evader = \
                self.step_4_1_1_preprocessing_observation(observations["evader"])

            actions_evader, values_evader, actions_log_probability_evader = \
                self.evader.model_actor_critic(observations_actor_evader, observations_critic_evader)

            observations_actor_pursuer, observations_critic_pursuer = \
                self.step_4_1_1_preprocessing_observation(observations["pursuer"])

            actions_pursuer, values_pursuer, actions_log_probability_pursuer = \
                self.pursuer.model_actor_critic(observations_actor_pursuer, observations_critic_pursuer)

            # 2. Env update and get observation.

            next_observations, rewards, dones, info = self.env.step(actions_pursuer=actions_pursuer,
                                                                    actions_evader=actions_evader)

            # 3. Update episode process record.

            self.episode_timestep_counter += 1

            self.logger.update_episode_additive_performance(key="reward_evader", value=np.mean(rewards["evader"]))
            self.logger.update_episode_additive_performance(key="n_collisions_evaders_with_obstacles",
                                                            value=info["n_collisions_evaders_with_obstacles"])
            self.logger.update_episode_additive_performance(key="n_collisions_evaders_with_pursuers",
                                                            value=info["n_collisions_evaders_with_pursuers"])
            self.logger.update_episode_additive_performance(key="n_collisions_evaders_with_evaders",
                                                            value=info["n_collisions_evaders_with_evaders"])

            # 4. Store agents' experience in buffer.

            self.evader.buffer.store(swarm_observations_actor=observations_actor_evader,
                                     swarm_observations_critic=observations_critic_evader,
                                     swarm_actions=actions_evader,
                                     swarm_rewards=torch.tensor(rewards["evader"], device=self.args.device),
                                     swarm_values=values_evader,
                                     swarm_actions_log_probability=actions_log_probability_evader,
                                     swarm_agent_active_index=info["evaders_last_active_index"])

            # 5. Update the observation memory.

            observations = next_observations

            # 6. Identify the game status and post-processing if done.

            observations = self.step_4_1_2_identify_and_terminate_an_episode(timestep, dones, observations, info)

            pass

    def step_4_1_1_preprocessing_observation(self, observations):

        # (n_agents, fov, fov, 3).

        observations_actor = torch.as_tensor(observations, dtype=torch.double, device=self.args.device)

        # (n_agents, fov * fov * 3).

        flatten_dim = observations.shape[-3] * observations.shape[-2] * observations.shape[-1]

        n_dim = len(observations.shape)

        if n_dim > 4:

            observations = observations.reshape(*observations.shape[:-3], flatten_dim)

        elif n_dim == 4:

            observations = observations.reshape(observations.shape[0], flatten_dim)

        else:

            observations = observations.reshape(flatten_dim)

        observations_critic = observations
        observations_critic = torch.as_tensor(observations_critic, dtype=torch.double, device=self.args.device)

        return observations_actor, observations_critic

    def step_4_1_2_identify_and_terminate_an_episode(self, timestep, dones, observations, info):

        # 6. Identify the game status.

        episode_timeout = (self.episode_timestep_counter == self.args.max_episode_length)
        episode_terminal = dones["game_done"] or episode_timeout

        # Specific number of experiences in an epoch have already been collected,
        # terminate this epoch (and prepare to start the next one).

        epoch_ended = (timestep == (self.args.steps_per_epoch - 1))

        # 7. Finish the experience collecting process if conditions are satisfied.

        for idx_evader, done_evader in enumerate(dones["evader"]):

            last_global_idx_active_evader = info["evaders_last_active_index"][idx_evader]

            if done_evader:

                last_value_evader = torch.zeros((1,), dtype=torch.double, device=self.args.device)

                self.evader.buffer.finish_path_by_index(last_value_evader, last_global_idx_active_evader)

            elif episode_timeout or epoch_ended:

                idx_active_evader = info["evaders_active_index"].tolist().index(last_global_idx_active_evader)

                observation_actor_evader, observation_critic_evader = \
                    self.step_4_1_1_preprocessing_observation(observations["evader"][idx_active_evader])

                _, last_value_evader, _ = self.evader.model_actor_critic(observation_actor_evader,
                                                                         observation_critic_evader)

                self.evader.buffer.finish_path_by_index(last_value_evader, last_global_idx_active_evader)

        if episode_terminal or epoch_ended:

            if self.args.render:
                self.env.render()

            # 9. When a complete episode is finished, log info and reset episode logger.
            # Otherwise, just reset the episode performance logger.

            if episode_terminal:

                self.logger.update_episode_additive_performance(key="capture_rate", value=info["capture_rate"])
                self.logger.update_episode_additive_performance(key="episode_length",
                                                                value=self.episode_timestep_counter)
                self.logger.end_episode_additive_performance()

            else:

                self.logger.reset_episode_additive_performance()

            # Reset.

            self.episode_timestep_counter = 0

            observations, rewards, game_done, info = self.env.reset()

        return observations

    def step_4_2_update_model(self):

        self.loss_policy_evader, self.loss_value_evader = self.evader.update_model()

        self.logger.update_epoch_performance(key="loss_value_evader", value=self.loss_value_evader)
        self.logger.update_epoch_performance(key="loss_policy_evader", value=self.loss_policy_evader)

        if (self.epoch_counter % self. args.save_model_frequency == 0) or \
                (self.epoch_counter == self.args.n_epochs - 1):

            model_name_prefix = os.path.join(self.args.data_log_folder,
                                             "Epoch" + str(self.args.model_name_frequency) + "x" +
                                             str(self.epoch_counter // self.args.model_name_frequency))
            model_name_actor_prefix = model_name_prefix + "model_actor"
            model_name_critic_prefix = model_name_prefix + "model_critic"
            model_filename_actor_evader = model_name_actor_prefix + "_evader.pth"
            model_filename_critic_evader = model_name_critic_prefix + "_evader.pth"

            torch.save(self.evader.model_actor_critic.model_policy.state_dict(), model_filename_actor_evader)
            torch.save(self.evader.model_actor_critic.model_value.state_dict(), model_filename_critic_evader)

    def step_4_3_log_info_of_epoch(self, epoch_time):

        self.logger.update_epoch_performance(key="epoch_time_s", value=epoch_time)

        if (self.epoch_counter % self.args.log_performance_frequency == 0) or \
                (self.epoch_counter == self.args.n_epochs - 1):

            self.logger.log_dump_epoch_performance(epoch_counter=self.epoch_counter)

        else:

            self.logger.reset_epoch_performance()


if __name__ == "__main__":

    all_args = parse_args()

    run_an_experiment = RunExperiment(all_args)

    run_an_experiment.run()

    print("COMPLETE!")

