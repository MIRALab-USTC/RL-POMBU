if __name__ == "__main__":
    import sys
    sys.path.insert(0,"/home/qizhou/libs/gym")
    sys.path.insert(0,"/home/qizhou/aaai2019")
    sys.path.insert(0,"/home/qzhou/aaai2019")
    sys.path.insert(0,"/home/qizhou/aaai2019/POMBU")
    sys.path.insert(0,"/home/qzhou/aaai2019/POMBU")
    print(sys.path)

import argparse
import numpy as np
import os
import time

from POMBU.train import train
from POMBU.utils import dump_params_to_file, load_params_from_file, eval_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run experiment options')
    parser.add_argument('-env_name')
    parser.add_argument('-algo')
    parser.add_argument('-seed')
    parser.add_argument('-exp_name')
    parser.add_argument('-subexp_name')
    parser.add_argument('-parent_dir')
    parser.add_argument('-report')

    running_kwargs = vars(parser.parse_args())
    
    if running_kwargs["exp_name"] == None:
        running_kwargs["exp_name"] = running_kwargs["env_name"]
        
    train_kwargs = load_params_from_file(running_kwargs["exp_name"], running_kwargs["subexp_name"])

    for key in running_kwargs:
        if key in train_kwargs["running_kwargs"] and running_kwargs[key] != None:
            train_kwargs["running_kwargs"][key] = running_kwargs[key]
    if running_kwargs["report"] != None:
        if running_kwargs["report"]:
            train_kwargs["running_kwargs"]["report_return_during_iteration"] = True
        else:
            train_kwargs["running_kwargs"]["report_return_during_iteration"] = False


    seed = train_kwargs["running_kwargs"]["seed"]
    if seed == "random":
        seed = train_kwargs["running_kwargs"]["seed"] = np.random.randint(0,4294967294)
    else:
        seed = train_kwargs["running_kwargs"]["seed"] = int(seed) % 4294967294
        
    parent_dir = train_kwargs["running_kwargs"]["parent_dir"]
    exp_name = train_kwargs["running_kwargs"]["exp_name"]
    subexp_name = train_kwargs["running_kwargs"]["subexp_name"] if "subexp_name" in train_kwargs["running_kwargs"] else None
    
    if subexp_name == "" or subexp_name == None: 
        save_dir = os.path.join(parent_dir, exp_name, "seed%d"%seed)
    else:
        save_dir = os.path.join(parent_dir, exp_name, subexp_name, "seed%d"%seed)

    train_kwargs["running_kwargs"]["save_dir"] = save_dir
    dump_params_to_file(train_kwargs, save_dir)
    print(train_kwargs)
    train_kwargs = eval_dict(train_kwargs)
    ts = time.time()
    train(train_kwargs)
    te = time.time()
    print("total time:%f"%(te - ts))
