import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


plt.style.use('ggplot')
import matplotlib
matplotlib.use('TkAgg')

SMALL_SIZE = 30
MEDIUM_SIZE = 35
BIGGER_SIZE = 40
LINEWIDTH = 4

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

colors = ["orangered",  "lightseagreen", "cornflowerblue", "orchid", "orange", "darkseagreen", "darkgray", "darkorange", "goldenrod", "mediumorchid", "darkturquoise" ]
COLORS = {
    "alpha=0":"orange",
    "alpha=0.25":"orchid",
    "alpha=0.5":"orangered",
    "alpha=0.75":"cornflowerblue",
    "alpha=1":"lightseagreen",
}
LEGEND_ORDER = {
    "alpha=0":0,
    "alpha=0.25":1,
    "alpha=0.5":2,
    "alpha=0.75":3,
    "alpha=1":4,
}



def get_color(label):
    if label not in COLORS.keys():
        COLORS[label] = colors.pop(0)
    return COLORS[label]


def get_plot_data_from_single_experiment(file_name, value_type):
    try:
        data = pd.read_csv(file_name)
    except pd.errors.EmptyDataError:
        return None, None
    
    column_y = "Value"
    column_x = "Step"

    data_y = list(data[column_y])
    data_x = list(data[column_x])

    if value_type in {"kl", "p_improve", "penalty"}:
        iter_count = 1
        sum_value = 0
        step_count = 0
        _data_x = []
        _data_y = []
        for i in range(len(data_x)):
            step = data_x[i]
            value = data_y[i]
            while step >= (iter_count * 20):
                if step_count != 0:
                    _data_x.append(iter_count)
                    _data_y.append(sum_value/step_count)
                sum_value = 0
                step_count = 0
                iter_count += 1
            step_count += 1
            sum_value += value
        data_x = _data_x
        data_y = _data_y
    elif value_type == "entropy":
        iter_count = 1
        _data_x = []
        _data_y = []
        data_size = len(data_x)
        for i in range(data_size):
            step = data_x[i]
            if step < iter_count*20 and (i==(data_size-1) or data_x[i+1] >= iter_count*20):
                value = data_y[i]
                _data_x.append(iter_count)
                _data_y.append(value)
                iter_count += 1
        data_x = _data_x
        data_y = _data_y
    else:
        for i in range(len(data_x)):
            data_x[i] = data_x[i] + 1
    return np.array(data_x).astype(np.int), np.array(data_y).astype(np.float)

def get_data_from_value_dir(value_dir, value_type):
    plot_y = []
    plot_x = []
    for sub_name in os.listdir(value_dir):
        file_name = os.path.join(value_dir, sub_name)
        if ".csv" in file_name:
            print("Obtaining data from %s"%file_name)
            x, y = get_plot_data_from_single_experiment(file_name, value_type)
            plot_x.append(x)
            plot_y.append(y)
    if plot_x == []:
        return None, None, None, None
    x, y_mean, y_std, max_y = compute_mean_std_max(plot_x, plot_y)
    return x, y_mean, y_std, max_y


def compute_mean_std_max(plot_x, plot_y):
    max_len = len(plot_y[0])
    x = plot_x[0]
    for i in range(1, len(plot_y)):
        if len(plot_y[i]) > max_len:
            max_len = len(plot_y[i])
            x = plot_x[i]
    y_mean = []
    y_std = []
    for itr in range(max_len):
        itr_values = []
        for curve_i in range(len(plot_y)):
            if itr < len(plot_y[curve_i]):
                itr_values.append(plot_y[curve_i][itr])
        y_mean.append(np.mean(itr_values))
        y_std.append(np.std(itr_values))

    y_mean = np.array(y_mean)
    y_std = np.array(y_std)
    return x, y_mean, y_std, np.max(y_mean)

def add_legend(fig, axs):
    _handles = []
    _labels = []
    for ax in axs:
        handles, labels = ax.get_legend_handles_labels()
        _handles += handles
        _labels += labels
    unique = [(h, l) for i, (h, l) in enumerate(zip(_handles, _labels)) if l not in _labels[:i]]
    handles, labels = zip(*unique)
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_transform=plt.gcf().transFigure)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training curve with progress data")
    parser.add_argument("--log_dir", type=str, default="local_data\\cheetah")
    parser.add_argument("--fig_name", type=str, default="verification.png")
    parser.add_argument("--w", type=float, default=34)
    parser.add_argument("--h", type=float, default=18)

    options = parser.parse_args()
    
    fig, axs = plt.subplots(2, 3, figsize=(options.w, options.h))
    axs = [axs[0,0], axs[0,1], axs[0,2], axs[1,0], axs[1,1], axs[1,2]]
    i=0
    log_dir = options.log_dir
    for value_type in os.listdir(log_dir):
        value_dir = os.path.join(log_dir, value_type)
        if not os.path.isdir(value_dir):
            continue
        ax = axs[i]
        sub_figure_name = value_type
        print("plot: %s" % sub_figure_name)
        ax.set_title(sub_figure_name)
        plot_data = []
        for alpha in os.listdir(value_dir):
            alpha_dir = os.path.join(value_dir, alpha)
            x, y_mean, y_std, max_y = get_data_from_value_dir(alpha_dir, value_type)
            plot_data.append((LEGEND_ORDER[alpha], alpha, x, y_mean, y_std, max_y))
        for _, alpha, x, y_mean, y_std, max_y in sorted(plot_data, key=lambda x: x[0]):
            if type(x) != type(None):
                ax.plot(x, y_mean, label=alpha, linewidth=LINEWIDTH, color=get_color(alpha))
                ax.fill_between(x, y_mean + y_std, y_mean - y_std, alpha=0.2, color=get_color(alpha))
                ax.set_xlim(0, 100)
                print("DONE:%s"%alpha)
            else:
                print("FAILED:%s"%alpha)
        
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value")
        i += 1

    add_legend(fig, axs)
    plt.tight_layout(pad=4.0, w_pad=1.5, h_pad=3, rect=[0, 0, 1, 1])
    #fig.savefig(os.path.join(log_dir, options.fig_name))
    print(COLORS)
    plt.savefig(os.path.join(log_dir, options.fig_name))


