exp_name="cheetah_long"
env_name="half_cheetah_long"
run_file="/home/qzhou/aaai2019/POMBU/run.py"
subexp_name=("vc_kl_0dot75")
parent_dir="/home/qzhou/aaai_data"


CUDA_VISIBLE_DEVICES=5 python $run_file -env_name $env_name -exp_name $exp_name -subexp_name ${subexp_name[1]} -parent_dir $parent_dir -seed 45678 > /dev/null 2>&1 &
