import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


plt.style.use('ggplot')
import matplotlib
matplotlib.use('Agg')

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

colors = ["orangered", "darkorange", "lightseagreen", "cornflowerblue", "orchid", "darkgray", "darkseagreen", "goldenrod", "darkorange", "mediumorchid", "darkturquoise" ]
COLORS = dict()
LEGEND_ORDER = dict(POMBU=0, SLBO=1, METRPO=2, MEPPO=3, SAC=4, TD3=5, PPO=6)

algo_dic = {
        "metrpo": "METRPO",
        "mbppo": "MEPPO",
        "pombu_vc": "POMBU",
        #"pombu_vc": "POMBU_vc",
        "slbo": "SLBO",
        "sac": "SAC",
        #"td3": "TD3",
        "ppo": "PPO",
}



def get_color(label):
    if label not in COLORS.keys():
        COLORS[label] = colors.pop(0)
    return COLORS[label]


def get_plot_data_from_single_experiment(file_name, algo_name):
    try:
        data = pd.read_csv(file_name)
    except pd.errors.EmptyDataError:
        return None, None
    
    dic_y = {
        "METRPO": "AverageReturn",
        "MEPPO": "AverageReturn",
        "SLBO": "Real Env",
        "POMBU": "AverageReturn",
        #"POMBU_vc": "AverageReturn", 
        "SAC": "evaluation/Average Returns",
        "TD3": "evaluation/Average Returns",
        "PPO": "Value",
    }

    dic_x = {
        "METRPO": "TotalSamples",
        "MEPPO": "TotalSteps",
        "SLBO": "TotalSteps",
        "POMBU": "TotalSteps",
        #"POMBU_vc": "TotalSteps", 
        "SAC": "exploration/num steps total",
        "TD3": "exploration/num steps total",
        "PPO": "Step",
    }
    column_y = dic_y[algo_name]
    column_x = dic_x[algo_name]

    data_y = data[column_y]
    data_x = data[column_x]

    if algo_name == "SLBO":
        _data_x = []
        _data_y = []
        for i in range(len(data_x)):
            if i%2 == 1:
                _data_x.append((i+1)/2*10000)
                _data_y.append(data_y[i])
        data_x = _data_x
        data_y = _data_y
    total_step = list(data_x)[-1]
    interval = total_step / options.n_data
    _data_x = []
    _data_y = []
    last_step = 0
    for i in range(len(data_x)):
        if data_x[i] > last_step-1:
            _data_x.append(data_x[i])
            _data_y.append(data_y[i])
            last_step += interval 
    
    return np.array(_data_x).astype(np.int), np.array(_data_y).astype(np.float)

def get_data_from_algo_dir(algo_dir, algo_name, x_limit):
    plot_y = []
    plot_x = []
    for sub_name in os.listdir(algo_dir):
        if ".csv" in sub_name:
            file_name = os.path.join(algo_dir, sub_name)
        else:
            sub_dir = os.path.join(algo_dir, sub_name)
            file_name = os.path.join(sub_dir, options.file_name)
        if os.path.exists(file_name):
            print("Obtaining data from %s"%file_name)
            x, y = get_plot_data_from_single_experiment(file_name, algo_name)
            if type(x) != type(None):
                max_step = x[-1]
                print("max step:%d"%max_step)
                if max_step >= x_limit:
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

    



def env_name_to_figure_name(env_name):
    dic = {
        "ant": "Ant",
        "walker2d": "Walker2D",
        "swimmer": "Swimmer",
        "cheetah": "Cheetah",
    }
    return dic[env_name] if env_name in dic else env_name


def get_x_limit(env_name):
    dic = {
        "ant": 600000,
        "walker2d": 400000,
        "swimmer": 100000,
        "cheetah": 200000,
    }
    return dic[env_name] if env_name in dic else 200000

def add_legend(fig, axs):
    _handles = []
    _labels = []
    for ax in axs:
        handles, labels = ax.get_legend_handles_labels()
        _handles += handles
        _labels += labels
    unique = [(h, l) for i, (h, l) in enumerate(zip(_handles, _labels)) if l not in _labels[:i]]
    handles, labels = zip(*unique)
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_transform=plt.gcf().transFigure)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training curve with progress data")
    parser.add_argument("--log_dir", type=str, default="/home/qzhou/data_copy/comparison")
    parser.add_argument("--file_name", type=str, default="progress.csv")
    parser.add_argument("--column_x", type=str, default="Time steps")
    parser.add_argument("--column_y", type=str, default="Average return")
    parser.add_argument("--fig_name", type=str, default="comparison_plot.png")
    parser.add_argument("--n_data", type=int, default=80)
    parser.add_argument("--w", type=float, default=25)
    parser.add_argument("--h", type=float, default=20)

    options = parser.parse_args()
    fig = plt.figure(figsize=(options.w,options.h))
    gs = GridSpec(6, 4)
    ax1 = plt.subplot(gs[:3, :2])
    ax2 = plt.subplot(gs[:3, 2:])
    ax3 = plt.subplot(gs[3:, :2])
    ax4 = plt.subplot(gs[3:, 2:])
    axs = [ax1, ax2, ax3, ax4]
    
    #fig, axs = plt.subplots(2, 2, figsize=(20, 15))
    #axs = [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]
    #fig.tight_layout()
    i=0
    log_dir = options.log_dir
    for env_name in os.listdir(log_dir):
        env_dir = os.path.join(log_dir, env_name)
        if not os.path.isdir(env_dir):
            continue
        ax = axs[i]
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        sub_figure_name = env_name_to_figure_name(env_name)
        print("plot: %s" % sub_figure_name)
        x_limit = get_x_limit(env_name)
        ax.set_title(sub_figure_name)
        plot_data = []
        for algo_name in os.listdir(env_dir):
            algo_dir = os.path.join(env_dir, algo_name)
            if algo_name in algo_dic:
                algo_name = algo_dic[algo_name]
            else:
                continue
            x, y_mean, y_std, max_y = get_data_from_algo_dir(algo_dir, algo_name, x_limit)
            plot_data.append((LEGEND_ORDER[algo_name], algo_name, x, y_mean, y_std, max_y))
        for _, algo_name, x, y_mean, y_std, max_y in sorted(plot_data, key=lambda x: x[0]):
            print("algo_name: %s"%algo_name)
            if type(x) != type(None):
                ax.plot(x, y_mean, label=algo_name, linewidth=LINEWIDTH, color=get_color(algo_name))
                ax.fill_between(x, y_mean + y_std, y_mean - y_std, alpha=0.2, color=get_color(algo_name))
                ax.set_xlim(0, x_limit)
                if env_name == "swimmer":   
                    ax.set_ylim(ymin=-5)
                if algo_name == "SAC":
                    max_sac = max_y
                if algo_name == "PPO":
                    max_ppo = max_y
                print("DONE:%s"%algo_name)
            else:
                print("FAILED:%s"%algo_name)
        ax.plot(ax.get_xlim(), [max_sac]*2, 'k--', label="sac_max")
        ax.plot(ax.get_xlim(), [max_ppo]*2, 'k-.', label="ppo_max")
        ax.set_xlabel(options.column_x)
        ax.set_ylabel(options.column_y)
        i += 1

    add_legend(fig, axs)
    plt.tight_layout(pad=4.0, w_pad=1.5, h_pad=3, rect=[0, 0, 1, 1])
    #fig.savefig(os.path.join(log_dir, options.fig_name))
    print(COLORS)
    plt.savefig(os.path.join(log_dir, options.fig_name))


