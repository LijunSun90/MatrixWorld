import os
import time
import argparse
import copy
import imageio

import random
import numpy as np

import torch

from lib.environment.pursuit_evasion_o import MatrixWorld as Environment

from lib.utils.experiment_logger import ExperimentLogger
from lib.utils.models import ModelPolicySafe
from lib.utils.models import ModelPolicy


def parse_args(idx_generation_pursuer, idx_generation_evader):

    parser = argparse.ArgumentParser()

    ##################################################
    # Log parameters.

    model_folder = os.path.join("data", "o_train_pursuit_evasion_2_longer")
    os.makedirs(model_folder, exist_ok=True)
    parser.add_argument("--model_folder", type=str, default=model_folder)

    policy_model_name_pursuer = "generation_" + str(idx_generation_pursuer) + "_model_actor_pursuer.pth"
    parser.add_argument("--policy_model_name_pursuer", type=str,
                        default=os.path.join(model_folder, policy_model_name_pursuer))

    policy_model_name_evader = "generation_" + str(idx_generation_evader) + "_model_actor_evader.pth"
    parser.add_argument("--policy_model_name_evader", type=str,
                        default=os.path.join(model_folder, policy_model_name_evader))

    # data_log_folder = os.path.join(model_folder, "evaluate_adversarial_policy")
    data_log_folder = os.path.join(model_folder, "render_adversarial_policy")
    os.makedirs(data_log_folder, exist_ok=True)
    parser.add_argument("--data_log_folder", type=str, default=data_log_folder)

    adversarial_postfix = "pursuer_" + str(idx_generation_pursuer) + "_vs_evader_" + str(idx_generation_evader)

    frames_folder = os.path.join(data_log_folder, "frames_" + adversarial_postfix)
    parser.add_argument("--frames_folder", type=str, default=frames_folder)
    # os.makedirs(frames_folder, exist_ok=True)

    video_filename = "video_" + adversarial_postfix + ".gif"
    parser.add_argument("--video_filename", type=str, default=os.path.join(data_log_folder, video_filename))

    performance_log_filename = "experiment_" + adversarial_postfix + ".txt"
    parser.add_argument("--performance_log_filename", type=str,
                        default=os.path.join(data_log_folder, performance_log_filename))

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
    # Environment parameters.

    parser.add_argument("--world_size", type=int, default=40)
    parser.add_argument("--n_pursuers", type=int, default=8)
    parser.add_argument("--n_evaders", type=int, default=30)

    ##################################################
    # General experiment parameters.

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--render", type=bool, default=True)

    parser.add_argument("--render_display", type=bool, default=False)

    parser.add_argument("--render_save", type=bool, default=True)

    parser.add_argument("--n_epochs", type=int, default=1, help="Statistical test mode: 10. Render mode: 1.")

    parser.add_argument("--max_episode_length", type=int, default=500)

    if torch.cuda.device_count() > 1:
        device = torch.device('cuda:1')
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    parser.add_argument("--device", type=torch.device, default=device)

    parser.add_argument("--log_performance_frequency", type=int, default=1,
                        help="time duration between successive log printing.")

    return parser.parse_args()


class RunExperiment:

    def __init__(self, args):

        self.args = args

        self.logger = ExperimentLogger(self.args.data_log_folder, self.args.performance_log_filename)

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

        # step_4_experiment_over_n_epochs

        self.epoch_counter = 0

        self.episode_timestep_counter = 0

        self.dim_observation_actor = 0

        pass

    def run(self):

        self.step_1_create_environment()

        self.step_2_set_random_seed()

        self.step_3_create_agent()

        self.step_4_experiment_over_n_epochs()

        self.step_5_log_statistics_info()

        self.step_6_png_to_gif()

    def step_1_create_environment(self):

        self.env = Environment(world_rows=self.args.world_size, world_columns=self.args.world_size,
                               n_evaders=self.args.n_evaders, n_pursuers=self.args.n_pursuers,
                               fov_scope=11,
                               max_env_cycles=self.args.max_episode_length,
                               save_path=self.args.frames_folder)

        self.dim_observation_actor = (self.env.fov_scope, self.env.fov_scope, 3)

        self.dim_action = self.env.n_actions

        if self.args.render_save:

            os.makedirs(self.args.frames_folder, exist_ok=True)

        pass

    def step_2_set_random_seed(self):
        """
        - First, create an environment instance.
        - Second, set random seed, including that for the environment.
        - Third, do all the other things.
        """
        seed = self.args.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.env.reset(seed=seed)

    def step_3_create_agent(self):

        # Use args parameter to initialize.

        self.evader = self.step_3_1_create_agent(is_evader=True)

        self.pursuer = self.step_3_1_create_agent(is_evader=False)

    def step_3_1_create_agent(self, is_evader=False):

        # Policy.

        if self.args.with_obstacle_avoidance_mask:

            model_policy = ModelPolicySafe(dim_input=self.dim_observation_actor,
                                           dim_output=self.dim_action,
                                           hidden_sizes=(400, 300),
                                           with_layer_normalization=self.args.with_layer_normalization,
                                           use_initialization=self.args.use_initialization,
                                           obstacle_channel=[2])

        else:

            model_policy = ModelPolicy(dim_input=self.dim_observation_actor,
                                       dim_output=self.dim_action,
                                       hidden_sizes=(400, 300),
                                       with_layer_normalization=self.args.with_layer_normalization,
                                       use_initialization=self.args.use_initialization)

        if is_evader:

            model_policy.load_state_dict(torch.load(self.args.policy_model_name_evader, map_location=self.args.device))

        else:

            model_policy.load_state_dict(torch.load(self.args.policy_model_name_pursuer, map_location=self.args.device))

        model_policy.to(self.args.device)

        return model_policy

    def step_4_experiment_over_n_epochs(self):

        for i_epoch in range(self.args.n_epochs):

            start_epoch_time = time.time()

            self.step_4_1_experiment_of_an_epoch()

            self.step_4_3_log_info_of_epoch(time.time() - start_epoch_time)

            # Update.

            self.epoch_counter += 1

            pass

    def step_4_1_experiment_of_an_epoch(self):

        observations, rewards, game_done, info = self.env.reset()

        for timestep in range(self.args.max_episode_length):

            if self.args.render:

                self.env.render(is_display=self.args.render_display, is_save=self.args.render_save)

            # 1. Agents make decisions.

            observations_actor_evader = self.step_4_1_1_preprocessing_observation(observations["evader"])

            actions_evader, actions_log_probability_evader = self.evader(observations_actor_evader)

            observations_actor_pursuer = self.step_4_1_1_preprocessing_observation(observations["pursuer"])

            actions_pursuer, actions_log_probability_pursuer = self.pursuer(observations_actor_pursuer)

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

            self.logger.update_episode_additive_performance(key="reward_pursuer", value=np.mean(rewards["pursuer"]))
            self.logger.update_episode_additive_performance(key="n_collisions_pursuers_with_obstacles",
                                                            value=info["n_collisions_pursuers_with_obstacles"])
            self.logger.update_episode_additive_performance(key="n_collisions_pursuers_with_pursuers",
                                                            value=info["n_collisions_pursuers_with_pursuers"])
            self.logger.update_episode_additive_performance(key="n_collisions_pursuers_with_evaders",
                                                            value=info["n_collisions_pursuers_with_evaders"])

            # 4. Store agents' experience in buffer.

            # 5. Update the observation memory.

            observations = next_observations

            # 6. Identify the game status and post-processing if done.

            observations, episode_terminal = \
                self.step_4_1_2_identify_and_terminate_an_episode(dones, observations, info)

            if episode_terminal:
                break

            pass

    def step_4_1_1_preprocessing_observation(self, observations):

        # (n_agents, fov, fov, 3).

        observations_actor = torch.as_tensor(observations, dtype=torch.double, device=self.args.device)

        return observations_actor

    def step_4_1_2_identify_and_terminate_an_episode(self, dones, observations, info):

        # 6. Identify the game status.

        episode_timeout = (self.episode_timestep_counter == self.args.max_episode_length)
        episode_terminal = dones["game_done"] or episode_timeout

        if episode_terminal:

            if self.args.render:

                self.env.render(is_display=self.args.render_display, is_save=self.args.render_save)

            # 7. Finish the experience collecting process.

            # 9. When a complete episode is finished, log info and reset episode logger.
            # Otherwise, just reset the episode performance logger.

            self.logger.update_episode_additive_performance(key="capture_rate", value=info["capture_rate"])
            self.logger.update_episode_additive_performance(key="episode_length",
                                                            value=self.episode_timestep_counter)
            self.logger.end_episode_additive_performance()

            # Reset.

            self.episode_timestep_counter = 0

            observations, rewards, game_done, info = self.env.reset()

        return observations, episode_terminal

    def step_4_3_log_info_of_epoch(self, epoch_time):

        self.logger.update_epoch_performance(key="epoch_time_s", value=epoch_time)

        if (self.epoch_counter % self.args.log_performance_frequency == 0) or \
                (self.epoch_counter == self.args.n_epochs - 1):

            self.logger.log_dump_epoch_performance(epoch_counter=self.epoch_counter)

        else:

            self.logger.reset_epoch_performance()

    def step_5_log_statistics_info(self):

        self.logger.log_dump_statistics_of_all_epoch_performance()

    def step_6_png_to_gif(self):

        if not self.args.render:

            return

        command = "ffmpeg -i " + self.args.frames_folder + "/MatrixWorld%4d.png " + self.args.video_filename

        os.system(command)

        print("Write to:", self.args.video_filename)

        command = "rm -r " + self.args.frames_folder

        os.system(command)

        pass


def main():

    n_generations_pursuer = 30
    n_generations_evader = 30

    for idx_generation_pursuer in range(0, n_generations_pursuer + 1, 1):

        for idx_generation_evader in range(0, n_generations_evader + 1, 1):

            print("="*80)
            print("Generation: Pursuer", idx_generation_pursuer, "vs. Evader", idx_generation_evader)

            all_args = parse_args(idx_generation_pursuer=idx_generation_pursuer,
                                  idx_generation_evader=idx_generation_evader)

            run_an_experiment = RunExperiment(all_args)

            run_an_experiment.run()

    pass


def main_specific_competition():

    list_competitions = [[100, 100], [100, 99]]

    for idx_generation_pursuer, idx_generation_evader in list_competitions:

        print("=" * 80)
        print("Generation: Pursuer", idx_generation_pursuer, "vs. Evader", idx_generation_evader)

        all_args = parse_args(idx_generation_pursuer=idx_generation_pursuer,
                              idx_generation_evader=idx_generation_evader)

        run_an_experiment = RunExperiment(all_args)

        run_an_experiment.run()

    pass


def main_baseline():

    # 30 + 1 vs. -2

    n_generations_pursuer = 30
    idx_generation_evader = -2

    for idx_generation_pursuer in range(-1, n_generations_pursuer + 1, 1):

        print("=" * 80)
        print("Generation: Pursuer", idx_generation_pursuer, "vs. Evader", idx_generation_evader)

        all_args = parse_args(idx_generation_pursuer=idx_generation_pursuer,
                              idx_generation_evader=idx_generation_evader)

        run_an_experiment = RunExperiment(all_args)

        run_an_experiment.run()

    pass


if __name__ == "__main__":

    main()
    # main_baseline()
    # main_specific_competition()

    print("COMPLETE!")

