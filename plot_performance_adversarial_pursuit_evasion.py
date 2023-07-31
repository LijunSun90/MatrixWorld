import os
import argparse

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_log_folder", type=str,
                        default=os.path.join("data", "o_train_pursuit_evasion_1"))

    parser.add_argument("--data_log_filename", type=str, default="experiment.txt")

    parser.add_argument("--n_epoch_per_generation_pursuer", type=int, default=400)

    parser.add_argument("--n_epoch_per_generation_evader", type=int, default=400)

    parser.add_argument("--truncate_data", type=bool, default=False)
    parser.add_argument("--truncate_data_length", type=int, default=10000)

    return parser.parse_args()


def get_data(all_args):

    data_log_filename = os.path.join(all_args.data_log_folder, all_args.data_log_filename)

    data = pd.read_table(data_log_filename)

    if all_args.truncate_data:

        data = data.iloc[:all_args.truncate_data_length]

    return data


def compute_moving_average(data, window_size=30):  # 20
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


def plot_a_metric(x, y, column_name, all_args, is_save=False):

    plt.figure(figsize=(20, 10))

    # Data.

    # handle_data = plt.plot(x, y, 'dimgray', label="Learning curve")

    # Moving average.
    y_moving_average = compute_moving_average(y)
    handle_moving_average = plt.plot(x, y_moving_average, '-', color='forestgreen', linewidth=2, label="Moving average")

    # Deal with outlier and y axis range.

    y_min = y.min()
    y_max = y.max()
    y_lim_bottom, y_lim_top = y_min, y_max
    q1 = y.quantile(0.25)
    q3 = y.quantile(0.75)

    if q1 != q3:

        iqr_interquartile_range = q3 - q1
        outlier_lower_bound = max(q1 - iqr_interquartile_range * 1.5, y_min)
        outlier_upper_bound = min(q3 + iqr_interquartile_range * 1.5, y_max)

        n_y = len(y)
        percentage_of_outlier_below_lower_bound = (y <= outlier_lower_bound).sum() / n_y
        percentage_of_outlier_above_upper_bound = (y >= outlier_upper_bound).sum() / n_y

        y_lim_bottom = outlier_lower_bound if percentage_of_outlier_below_lower_bound < 0.15 else y_min
        y_lim_top = outlier_upper_bound if percentage_of_outlier_above_upper_bound < 0.15 else y_max

        plt.ylim([y_lim_bottom, y_lim_top])

    # Plot colored rectangle background.

    x_bar_1 = np.arange(start=all_args.n_epoch_per_generation_pursuer * 0.5, stop=len(x),
                        step=all_args.n_epoch_per_generation_pursuer + all_args.n_epoch_per_generation_evader)
    color_list_1 = ['b'] * len(x_bar_1)
    bar_1 = plt.bar(x_bar_1, height=y_lim_top - y_lim_bottom,
                    width=all_args.n_epoch_per_generation_pursuer, bottom=y_lim_bottom,
                    color=color_list_1, alpha=0.2)

    x_bar_2 = np.arange(start=all_args.n_epoch_per_generation_pursuer + all_args.n_epoch_per_generation_evader * 0.5,
                        stop=len(x),
                        step=all_args.n_epoch_per_generation_pursuer + all_args.n_epoch_per_generation_evader)
    color_list_2 = ['r'] * len(x_bar_2)
    bar_2 = plt.bar(x_bar_2, height=y_lim_top - y_lim_bottom,
                    width=all_args.n_epoch_per_generation_evader, bottom=y_lim_bottom,
                    color=color_list_2, alpha=0.2, label="Evader learn")

    # Labels.

    font_size = 30
    font_size_diff = 5

    # plt.legend()
    # The plot objects get wrapped in arrays. So, add an index [0] for legend.
    # plt.legend([bar_1, bar_2, handle_data[0], handle_moving_average[0]],
    #            ["Pursuer learn", "Evader learn", "Learning curve", "Moving average"],
    #            fontsize=font_size - font_size_diff)
    plt.legend([bar_1, bar_2, handle_moving_average[0]],
               ["Pursuer learn", "Evader learn", "Learning curve"],
               fontsize=font_size - font_size_diff)

    plt.xlabel("Epoch", fontsize=font_size)
    y_label = " ".join(column_name.split("_")).capitalize()
    plt.ylabel(y_label, fontsize=font_size)

    plt.xticks(fontsize=font_size - font_size_diff)
    plt.yticks(fontsize=font_size - font_size_diff)

    if is_save:

        figure_log_folder = os.path.join(all_args.data_log_folder, "log_performance_figures")
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

        if column_name != "capture_rate":
            continue

        plot_a_metric(x, data[column_name], column_name, all_args, is_save=True)

    plt.show()

    pass


if __name__ == "__main__":
    main()
    print("COMPLETE!")
