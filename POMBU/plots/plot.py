import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.style.use('ggplot')
import matplotlib
matplotlib.use('Agg')


def get_plot_data_from_single_experiment(exp_dir, file_name, column_x, column_y, start_index=1):
    try:
        data = pd.read_csv(os.path.join(exp_dir, file_name))
    except pd.errors.EmptyDataError:
        return [], []

    data_y = data[column_y]
    if options.column_y_transformation == "softplus":
        data_y = [np.log(np.exp(datum_y) + 1) for datum_y in data_y]

    if column_x == "Iteration":
        data_x = range(start_index, len(data_y) + start_index)
    else:
        data_x = data[column_x]
    return data_x, data_y


def add_curve_for_experiment(data_dir, label=None, start_index=1):
    plot_data = []
    plot_indices = []
    for sub_name in os.listdir(data_dir):
        sub_dir = os.path.join(data_dir, sub_name)
        if os.path.isdir(sub_dir) and "seed" in sub_name and os.path.exists(os.path.join(sub_dir, options.file_name)):
            print("=> Obtaining data from subdirectory %s"%sub_dir)
            plot_indices_single_exp, plot_data_single_exp = get_plot_data_from_single_experiment(
                sub_dir, options.file_name, options.column_x, options.column_y, start_index=start_index)
            plot_indices.append(plot_indices_single_exp)
            plot_data.append(plot_data_single_exp)
    add_curve(plot_data, plot_indices, label)


def create_plot():
    plt.figure()


def save_plot(exp_dir):
    plt.xlabel("Time steps")
    plt.ylabel("Average return")
    plt.savefig(os.path.join(exp_dir, options.save_fig_filename))
    plt.close()


def add_curve(data, indices, label):
    longest_length = len(data[0])
    longest_length_indices = indices[0]
    for i in range(1, len(data)):
        if len(data[i]) > longest_length:
            longest_length = len(data[i])
            longest_length_indices = indices[i]
    mean_data = []
    std_data = []
    for itr in range(longest_length):
        itr_values = []
        for curve_i in range(len(data)):
            if itr < len(data[curve_i]):
                itr_values.append(data[curve_i][itr])
        mean_data.append(np.mean(itr_values))
        std_data.append(np.std(itr_values))

    mean_data = np.array(mean_data)
    std_data = np.array(std_data)

    if label is None:
        plt.plot(longest_length_indices, mean_data)
    else:
        if label in COLORS:
            plt.plot(longest_length_indices, mean_data, label=label, color=COLORS[label])
        else:
            plt.plot(longest_length_indices, mean_data, label=label)
    if label in COLORS:
        plt.fill_between(longest_length_indices, mean_data + std_data, mean_data - std_data, alpha=0.3, color=COLORS[label])
    else:
        plt.fill_between(longest_length_indices, mean_data + std_data, mean_data - std_data, alpha=0.3)

def exptype_to_function(exp_type):
    def _transfer_name_to_label(name):
        alpha = sub_name[sub_name.rfind("_")+1:]
        dot = alpha.find("dot")
        if dot != -1:
            alpha = alpha[:dot] + "." + alpha[dot+3:]
        else:
            alpha = alpha
        return "alpha="+alpha
    def _transfer_name_to_label_beta(name):
        if "simple" in name:
            return "single"
        beta = sub_name[sub_name.rfind("_")+1:]
        dot = beta.find("dot")
        if dot != -1:
            beta = beta[:dot] + "." + beta[dot+3:]
        else:
            beta = beta
        return "beta="+beta

    dic = {
        "ablation": _transfer_name_to_label,
        "exploration": _transfer_name_to_label_beta,
    }
    return dic[exp_type] if exp_type in dic else lambda s: s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training curve with progress data")
    parser.add_argument("--parent_dir", type=str, default="/home/qzhou/data_copy/")
    parser.add_argument("--exp_name", type=str, default="ablation")
    parser.add_argument("--exp_type", type=str, default="ablation")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--file_name", type=str, default="progress.csv")
    parser.add_argument("--column_x", type=str, default="TotalSteps")
    parser.add_argument("--column_y", type=str, default="AverageReturn")
    parser.add_argument("--column_y_transformation", type=str, default=None)

    parser.add_argument("--save_fig_filename", type=str, default="plot.png")
    parser.add_argument("--x_max", type=int, default=200000)

    options = parser.parse_args()
    if options.exp_type=="exploration":
        COLORS = {"single":"deepskyblue", "beta=10":"salmon","beta=0":"deepskyblue"}
    elif options.exp_type=="ablation":
        COLORS = {
            "alpha=0":"orange",
            "alpha=0.25":"orchid",
            "alpha=0.5":"orangered",
            "alpha=0.75":"cornflowerblue",
            "alpha=1":"lightseagreen",
        }


    create_plot()
    sub_name_to_label = exptype_to_function(options.exp_type)
    exp_dir = os.path.join(options.parent_dir, options.exp_name)
    for sub_name in os.listdir(exp_dir):
        if "p_improve" in sub_name or sub_name=="alpha_0":
            continue
        sub_dir = os.path.join(exp_dir, sub_name)
        if os.path.isdir(sub_dir):
            print("current director: %s"%sub_dir)
            label = sub_name_to_label(sub_name)
            add_curve_for_experiment(sub_dir, label=label, start_index=options.start_index)
    plt.xlim(left=0, right=options.x_max)
    handles, labels = plt.gca().get_legend_handles_labels()

    key_dic = {
        "ablation": lambda item: eval(item[1][item[1].rfind("=")+1:]),
    }

    key = key_dic[options.exp_type] if options.exp_type in key_dic else lambda f: f[1]
    plt.legend((*zip(*sorted(zip(handles, labels),key=key))), loc="best")
    save_plot(exp_dir)


