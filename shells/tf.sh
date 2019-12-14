exp_name="op_method"
run_file="/home/qzhou/aaai2019/POMBU/run.py"
subexp_name=("vc_kl" "vc_kl_adaptive_alpha" "p_improve_kl_adaptive_clip" "p_improve_kl" "p_improve")
parent_dir="/home/qzhou/aaai_data"

tensorboard --logdir $parent_dir/$exp_name/${subexp_name[0]}/seed123456 --port 6600 > /dev/null 2>&1 &
tensorboard --logdir $parent_dir/$exp_name/${subexp_name[0]}/seed234567 --port 6601 > /dev/null 2>&1 &
tensorboard --logdir $parent_dir/$exp_name/${subexp_name[0]}/seed345678 --port 6602 > /dev/null 2>&1 &
tensorboard --logdir $parent_dir/$exp_name/${subexp_name[0]}/seed456789 --port 6603 > /dev/null 2>&1 &

tensorboard --logdir $parent_dir/$exp_name/${subexp_name[1]}/seed123456 --port 6604 > /dev/null 2>&1 &
tensorboard --logdir $parent_dir/$exp_name/${subexp_name[1]}/seed234567 --port 6605 > /dev/null 2>&1 &
tensorboard --logdir $parent_dir/$exp_name/${subexp_name[1]}/seed345678 --port 6606 > /dev/null 2>&1 &
tensorboard --logdir $parent_dir/$exp_name/${subexp_name[1]}/seed456789 --port 6607 > /dev/null 2>&1 &

tensorboard --logdir $parent_dir/$exp_name/${subexp_name[2]}/seed123456 --port 6608 > /dev/null 2>&1 &
tensorboard --logdir $parent_dir/$exp_name/${subexp_name[2]}/seed234567 --port 6609 > /dev/null 2>&1 &
tensorboard --logdir $parent_dir/$exp_name/${subexp_name[2]}/seed345678 --port 6610 > /dev/null 2>&1 &
tensorboard --logdir $parent_dir/$exp_name/${subexp_name[2]}/seed456789 --port 6611 > /dev/null 2>&1 &

tensorboard --logdir $parent_dir/$exp_name/${subexp_name[3]}/seed123456 --port 6612 > /dev/null 2>&1 &
tensorboard --logdir $parent_dir/$exp_name/${subexp_name[3]}/seed234567 --port 6613 > /dev/null 2>&1 &
tensorboard --logdir $parent_dir/$exp_name/${subexp_name[3]}/seed345678 --port 6614 > /dev/null 2>&1 &
tensorboard --logdir $parent_dir/$exp_name/${subexp_name[3]}/seed456789 --port 6615 > /dev/null 2>&1 &

tensorboard --logdir $parent_dir/$exp_name/${subexp_name[4]}/seed123456 --port 6616 > /dev/null 2>&1 &
tensorboard --logdir $parent_dir/$exp_name/${subexp_name[4]}/seed234567 --port 6617 > /dev/null 2>&1 &
tensorboard --logdir $parent_dir/$exp_name/${subexp_name[4]}/seed345678 --port 6618 > /dev/null 2>&1 &
tensorboard --logdir $parent_dir/$exp_name/${subexp_name[4]}/seed456789 --port 6619 > /dev/null 2>&1 &