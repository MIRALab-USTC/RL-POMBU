import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.style.use('ggplot')
import matplotlib
matplotlib.use('Agg')

COLORS = {"MBPPO":"deepskyblue", "POMBU":"salmon","METRPO":"orange"}
LABELED = {"MBPPO":False, "POMBU":False, "METRPO":False}
dic_xy = {
    "MBPPO":["TotalSteps", "AverageReturn"],
    "POMBU":["TotalSteps", "AverageReturn"],
    "METRPO":["TotalSamples", "AverageReturn"],
}

def get_color(label):
    if LABELED[label]:
        return COLORS[label], False
    else:
        LABELED[label] = True
        return COLORS[label], True

def plot_single_experiment(exp_dir, file_name, column_x, column_y, label, start_index=1):
    try:
        data = pd.read_csv(os.path.join(exp_dir, file_name))
    except pd.errors.EmptyDataError:
        return

    data_y = data[column_y]
    if options.column_y_transformation == "softplus":
        data_y = [np.log(np.exp(datum_y) + 1) for datum_y in data_y]

    if column_x == "Iteration":
        data_x = range(start_index, len(data_y) + start_index)
    else:
        data_x = data[column_x]
    
    for i in range(len(data_x)):
        if data_x[i] > options.x_max:
            break
    data_x = data_x[:i]
    data_y = data_y[:i]

    color, _label = get_color(label)
    if _label:
        plt.plot(data_x, data_y, label=label, color=color)
    else:
        plt.plot(data_x, data_y, label="", color=color)


def add_curve_for_experiment(data_dir, label=None, start_index=1):
    for sub_name in os.listdir(data_dir):
        sub_dir = os.path.join(data_dir, sub_name)
        if os.path.isdir(sub_dir) and "seed" in sub_name and os.path.exists(os.path.join(sub_dir, options.file_name)):
            print("=> Obtaining data from subdirectory %s"%sub_dir)
            plot_single_experiment(
                sub_dir, options.file_name, dic_xy[label][0], dic_xy[label][1], label, start_index=start_index)


def create_plot():
    return plt.figure()


def save_plot(exp_dir):
    plt.xlabel("Time steps")
    plt.ylabel("Average return")
    plt.savefig(os.path.join(exp_dir, options.save_fig_filename))
    plt.close()

def add_legend():
    plt.legend(loc='best')


def dirname_to_label(dir_name):
        if dir_name == "metrpo":
            return "METRPO"
        else:
            alpha = dir_name[dir_name.rfind("_")+1:]
            if alpha == "0":
                return "MBPPO"
            else:
                return "POMBU"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training curve with progress data")
    parser.add_argument("--parent_dir", type=str, default="/home/qzhou/data_copy/")
    parser.add_argument("--exp_name", type=str, default="noisy_O01")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--file_name", type=str, default="progress.csv")
    parser.add_argument("--column_x", type=str, default="TotalSteps")
    parser.add_argument("--column_y", type=str, default="AverageReturn")
    parser.add_argument("--column_y_transformation", type=str, default=None)

    parser.add_argument("--save_fig_filename", type=str, default="noise_plot.png")
    parser.add_argument("--x_max", type=int, default=200000)

    options = parser.parse_args()
    if options.exp_name == "exploration_beta":
        COLORS = {"single":"deepskyblue", "beta=10":"salmon","beta=0":"deepskyblue"}
        LABELED = {"single":False, "beta=10":False, "beta=0":False}
        dic_xy = {
            "single":["TotalSteps", "AverageReturn"],
            "beta=10":["TotalSteps", "AverageReturn"],
            "beta=0":["TotalSteps", "AverageReturn"],
        }
    if options.exp_name == "exploration_beta":
        def dirname_to_label(name):
            if "simple" in name:
                return "single"
            beta = sub_name[sub_name.rfind("_")+1:]
            dot = beta.find("dot")
            if dot != -1:
                beta = beta[:dot] + "." + beta[dot+3:]
            else:
                beta = beta
            return "beta="+beta


    fig = create_plot()
    exp_dir = os.path.join(options.parent_dir, options.exp_name)
    for sub_name in os.listdir(exp_dir):
        if "p_improve" in sub_name:
            continue
        if options.exp_name == "exploration_beta" and sub_name=="alpha_0":
            continue
        sub_dir = os.path.join(exp_dir, sub_name)
        if os.path.isdir(sub_dir):
            print("current director: %s"%sub_dir)
            label = dirname_to_label(sub_name)
            add_curve_for_experiment(sub_dir, label=label, start_index=options.start_index)
    if options.exp_name == "exploration_beta":
        plt.xlim(left=99000, right=options.x_max)
        plt.ylim(bottom=470, top=930)
    else:
        plt.xlim(left=0, right=options.x_max)
    add_legend()
    save_plot(exp_dir)


