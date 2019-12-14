exp_name="op_method_halfcheetah"
env_name="half_cheetah"
run_file="/home/qzhou/aaai2019/POMBU/run.py"
subexp_name="vc_kl_0dot5_exalpha0"
parent_dir="/home/qzhou/aaai_data"

#rm -rf $parent_dir/$exp_name

CUDA_VISIBLE_DEVICES=2 python $run_file -env_name $env_name -exp_name $exp_name -subexp_name $subexp_name -parent_dir $parent_dir -seed 12345 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=2 python $run_file -env_name $env_name -exp_name $exp_name -subexp_name $subexp_name -parent_dir $parent_dir -seed 23456 > /dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=2 python $run_file -env_name $env_name -exp_name $exp_name -subexp_name $subexp_name -parent_dir $parent_dir -seed 34567 &
CUDA_VISIBLE_DEVICES=2 python $run_file -env_name $env_name -exp_name $exp_name -subexp_name $subexp_name -parent_dir $parent_dir -seed 45678 > /dev/null 2>&1 &


