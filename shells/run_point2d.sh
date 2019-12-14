exp_name="point_2d"
env_name="point_2d"
run_file="/home/qzhou/aaai2019/POMBU/run.py"
subexp_name="point_2d"
parent_dir="/home/qzhou/aaai_data"
rm -rf $parent_dir/$exp_name

CUDA_VISIBLE_DEVICES=7 python $run_file -env_name $env_name -exp_name $exp_name -subexp_name $subexp_name -parent_dir $parent_dir -seed 12345 &
CUDA_VISIBLE_DEVICES=7 python $run_file -env_name $env_name -exp_name $exp_name -subexp_name $subexp_name -parent_dir $parent_dir -seed 23456 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=7 python $run_file -env_name $env_name -exp_name $exp_name -subexp_name $subexp_name -parent_dir $parent_dir -seed 34567 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=6 python $run_file -env_name $env_name -exp_name $exp_name -subexp_name $subexp_name -parent_dir $parent_dir -seed 45678 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=6 python $run_file -env_name $env_name -exp_name $exp_name -subexp_name $subexp_name -parent_dir $parent_dir -seed 56789 > /dev/null 2>&1 &
