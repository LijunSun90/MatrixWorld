"""
Author: Lijun Sun.
Date: Tue 21 Feb 2023.
"""
import os
import time
import numpy as np


class ExperimentLogger:
    """
    Function:
        - Multiple epochs.
        - Multiple episode per epoch.
        - Multiple timesteps per episode.
        - Record additive metric values over timesteps of an episode by update() every time step.
        - Record a single final metric value of an episode by update() once at the end of episode.
        - Log epoch performance as the mean of metric values over episodes per epoch.
    """

    def __init__(self, log_folder, log_file=None):

        self.log_folder = log_folder

        if log_file is None:

            # self.log_file = os.path.join(self.log_folder, "experiment" + time.strftime("%Y%m%d%H%M%S") + ".txt")
            self.log_file = os.path.join(self.log_folder, "experiment.txt")

            self.log_file_statistics = os.path.join(self.log_folder, "experiment_statistics.txt")

        else:

            self.log_file = log_file

            self.log_file_statistics = log_file[:-4] + "_statistics.txt"

        self.episode_additive_performance = dict()

        self.epoch_performance = dict()

        self.all_epoch_performance_statistics = dict(n_data=0)

        self.epoch_counter = 0

        self.log_row_head = ["epoch"]

        self.log_file_not_created = True

        pass

    def create_log_file(self):

        os.makedirs(self.log_folder, exist_ok=True)

        with open(self.log_file, 'w') as output_file:

            output_file.write("\t".join(map(str, self.log_row_head)) + "\n")
            output_file.flush()

        print("=> Create:", self.log_file)

    def reset_episode_additive_performance(self):

        for key in self.episode_additive_performance.keys():

            self.episode_additive_performance[key] = 0

    def reset_epoch_performance(self):

        for key in self.epoch_performance.keys():

            self.epoch_performance[key] = []

    def update_episode_additive_performance(self, key, value):

        if key in self.episode_additive_performance.keys():

            self.episode_additive_performance[key] += value

        else:

            self.episode_additive_performance[key] = value

            self.epoch_performance[key] = []

            self.log_row_head.append(key)

    def end_episode_additive_performance(self):
        """
        - Add episode performance to epoch performance.
        - Reset episode performance.
        """

        for key in self.episode_additive_performance.keys():

            self.epoch_performance[key].append(self.episode_additive_performance[key])

        # Update.

        self.reset_episode_additive_performance()

    def update_epoch_performance(self, key, value):

        if key in self.epoch_performance.keys():

            self.epoch_performance[key].append(value)

        else:

            self.epoch_performance[key] = [value]

            self.log_row_head.append(key)

    def log_dump_epoch_performance(self, epoch_counter=None, is_print=True):

        if epoch_counter is not None:
            self.epoch_counter = epoch_counter

        if self.log_file_not_created:

            self.log_file_not_created = False
            self.create_log_file()

        # Calculate mean performance per epoch.

        row = [self.epoch_counter]

        # For calculating statistics.

        self.all_epoch_performance_statistics["n_data"] += 1

        for key, value in self.epoch_performance.items():

            value = np.mean(value)

            row.append(value)

            # For calculating statistics.

            if key in self.all_epoch_performance_statistics.keys():

                self.all_epoch_performance_statistics[key].append(value)

            else:

                self.all_epoch_performance_statistics[key] = [value]

        # Write.

        with open(self.log_file, 'a') as output_file:

            output_file.write("\t".join(map(str, row)) + "\n")
            output_file.flush()

        # Print.

        if is_print:

            self.print_epoch_performance()

        # Update.

        if epoch_counter is None:

            self.epoch_counter += 1

        self.reset_epoch_performance()

        pass

    def print_epoch_performance(self):

        print("epoch: {:7d}".format(self.epoch_counter), end=", ")

        for key, value in self.epoch_performance.items():

            print(key + ": {:.5f}".format(np.mean(value)), end=", ")

        print()

    def log_dump_statistics_of_all_epoch_performance(self):

        row_head = []
        row_value = []

        for key, value in self.all_epoch_performance_statistics.items():

            if key == "n_data":

                row_head.append(key)
                row_value.append(value)

            else:

                row_head += [key + "_avg", key + "_std"]
                row_value += [np.mean(value), np.std(value)]

        all_rows = [row_head, row_value]

        # Write.

        with open(self.log_file_statistics, 'w') as output_file:

            for row in all_rows:

                output_file.write("\t".join(map(str, row)) + "\n")
                output_file.flush()

        print("Write to", self.log_file_statistics)

        # Print.

        for name, value in zip(row_head, row_value):

            print(name + ": {:.5f}".format(value), end=", ")

        print()

        pass


def example_usage():

    logger = ExperimentLogger("data")

    for idx_epoch in range(3):

        start_epoch_time = time.time()

        for idx_episode in range(5):

            for idx_timestep in range(10):

                logger.update_episode_additive_performance(key="episode_additive_metric", value=1.0)

            # Log info only when a complete episode is finished. Then, reset episode parameters.
            logger.update_episode_additive_performance(key="episode_final_metric", value=idx_episode)
            logger.end_episode_additive_performance()
            # Otherwise, just reset episode parameters.
            # logger.reset_episode_additive_performance()

        logger.update_epoch_performance(key="epoch_time_s", value=time.time() - start_epoch_time)

        logger.log_dump_epoch_performance()

    logger.log_dump_statistics_of_all_epoch_performance()

    pass


if __name__ == "__main__":

    example_usage()

    print('COMPLETE!')

