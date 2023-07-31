import os
import argparse

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_log_folder", type=str,
                        default=os.path.join("data", "s_train_pursuit_vs_random_evasion"))

    parser.add_argument("--data_log_filename", type=str, default="experiment.txt")

    parser.add_argument("--truncate_data", type=bool, default=False)
    parser.add_argument("--truncate_data_length", type=int, default=1000)

    return parser.parse_args()


def get_data(all_args):

    data_log_filename = os.path.join(all_args.data_log_folder, all_args.data_log_filename)

    data = pd.read_table(data_log_filename)

    if all_args.truncate_data:

        data = data.iloc[:all_args.truncate_data_length]

    return data


def compute_moving_average(data, window_size=30):
    """
    :param data: (n_data,).
    :param window_size: int.
    :return: (n_data,).

    Example behavior:

    data: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    moving_average: [1 1 1 3 3 3 3 3 3 3]

    data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    moving_average: [ 0  1  2  6  9 12 15 18 21 24]
    """

    data = np.array(data).squeeze()
    n_data = len(data)

    window_size = min(max(1, window_size), n_data)

    cumulative_sum = np.cumsum(data)
    intermediate_value = cumulative_sum[window_size:] - cumulative_sum[:-window_size]
    moving_average = np.hstack((data[:window_size], intermediate_value / window_size))

    return moving_average


def plot_a_metric(x, y, column_name, is_save=False, data_log_folder=None):

    plt.figure(figsize=(12, 5))

    plt.plot(x, y)

    y_moving_average = compute_moving_average(y)
    plt.plot(x, y_moving_average, 'r-.', label="Moving average")

    # Labels.

    font_size = 30
    font_size_diff = 5

    plt.legend(fontsize=font_size - font_size_diff)
    plt.xlabel("Epoch", fontsize=font_size)
    y_lablel = " ".join(column_name.split("_")).capitalize()
    plt.ylabel(y_lablel, fontsize=font_size)
    # plt.title("Collisions between pursuers and pursuers", fontsize=font_size)

    plt.xticks(fontsize=font_size - font_size_diff)
    plt.yticks(fontsize=font_size - font_size_diff)

    if is_save:

        figure_log_folder = os.path.join(data_log_folder, "log_performance_figures")
        os.makedirs(figure_log_folder, exist_ok=True)

        output_figure_path = os.path.join(figure_log_folder, column_name + ".png")
        plt.savefig(output_figure_path, bbox_inches='tight')
        print("Save to:", output_figure_path)

    pass


def main():

    all_args = parse_args()

    data = get_data(all_args)

    x = data["epoch"]

    for column_name in data.columns[1:]:
        # if column_name != "n_collisions_pursuers_with_pursuers":
        #     continue
        plot_a_metric(x, data[column_name], column_name, is_save=True, data_log_folder=all_args.data_log_folder)

    plt.show()
    pass


if __name__ == "__main__":
    main()
    print("COMPLETE!")
