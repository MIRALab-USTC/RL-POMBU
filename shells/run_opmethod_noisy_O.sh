exp_name=("noisy_O001" "noisy_O01")
env_name=("gym_cheetahO001" "gym_cheetahO01")
run_file="/home/qzhou/aaai2019/POMBU/run.py"
subexp_name=("vc_kl_0" "vc_kl_0dot5")
parent_dir="/home/qzhou/aaai_data"

rm -rf $parent_dir/$exp_name

CUDA_VISIBLE_DEVICES=2 python $run_file -env_name ${env_name[0]} -exp_name ${exp_name[0]} -subexp_name ${subexp_name[0]} -parent_dir $parent_dir -seed 12345 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=2 python $run_file -env_name ${env_name[0]} -exp_name ${exp_name[0]}  -subexp_name ${subexp_name[0]} -parent_dir $parent_dir -seed 23456 > /dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=2 python $run_file -env_name ${env_name[0]} -exp_name ${exp_name[0]}  -subexp_name ${subexp_name[0]} -parent_dir $parent_dir -seed 34567 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=3 python $run_file -env_name ${env_name[0]} -exp_name ${exp_name[0]}  -subexp_name ${subexp_name[0]} -parent_dir $parent_dir -seed 45678 &


CUDA_VISIBLE_DEVICES=3 python $run_file -env_name ${env_name[1]} -exp_name ${exp_name[1]}  -subexp_name ${subexp_name[1]} -parent_dir $parent_dir -seed 12345 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=3 python $run_file -env_name ${env_name[1]} -exp_name ${exp_name[1]}  -subexp_name ${subexp_name[1]} -parent_dir $parent_dir -seed 23456 > /dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=3 python $run_file -env_name ${env_name[1]} -exp_name ${exp_name[1]}  -subexp_name ${subexp_name[1]} -parent_dir $parent_dir -seed 34567 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=4 python $run_file -env_name ${env_name[1]} -exp_name ${exp_name[1]}  -subexp_name ${subexp_name[1]} -parent_dir $parent_dir -seed 45678 > /dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=4 python $run_file -env_name ${env_name[1]} -exp_name ${exp_name[1]}  -subexp_name ${subexp_name[0]} -parent_dir $parent_dir -seed 12345 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=4 python $run_file -env_name ${env_name[1]} -exp_name ${exp_name[1]}  -subexp_name ${subexp_name[0]} -parent_dir $parent_dir -seed 23456 > /dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=5 python $run_file -env_name ${env_name[1]} -exp_name ${exp_name[1]}  -subexp_name ${subexp_name[0]} -parent_dir $parent_dir -seed 34567 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=5 python $run_file -env_name ${env_name[1]} -exp_name ${exp_name[1]}  -subexp_name ${subexp_name[0]} -parent_dir $parent_dir -seed 45678 > /dev/null 2>&1 &

#CUDA_VISIBLE_DEVICES=5 python $run_file -env_name ant -exp_name $exp_name -subexp_name ${subexp_name[3]} -parent_dir $parent_dir -seed 12345 > /dev/null 2>&1 &
#CUDA_VISIBLE_DEVICES=5 python $run_file -env_name ant -exp_name $exp_name -subexp_name ${subexp_name[3]} -parent_dir $parent_dir -seed 23456 > /dev/null 2>&1 &

#CUDA_VISIBLE_DEVICES=5 python $run_file -env_name ant -exp_name $exp_name -subexp_name ${subexp_name[3]} -parent_dir $parent_dir -seed 34567 > /dev/null 2>&1 &
#CUDA_VISIBLE_DEVICES=6 python $run_file -env_name ant -exp_name $exp_name -subexp_name ${subexp_name[3]} -parent_dir $parent_dir -seed 45678 > /dev/null 2>&1 &

#CUDA_VISIBLE_DEVICES=6 python $run_file -env_name ant -exp_name $exp_name -subexp_name ${subexp_name[4]} -parent_dir $parent_dir -seed 12345 > /dev/null 2>&1 &
#CUDA_VISIBLE_DEVICES=6 python $run_file -env_name ant -exp_name $exp_name -subexp_name ${subexp_name[4]} -parent_dir $parent_dir -seed 23456 > /dev/null 2>&1 &

#CUDA_VISIBLE_DEVICES=6 python $run_file -env_name ant -exp_name $exp_name -subexp_name ${subexp_name[4]} -parent_dir $parent_dir -seed 34567 > /dev/null 2>&1 &
#CUDA_VISIBLE_DEVICES=6 python $run_file -env_name ant -exp_name $exp_name -subexp_name ${subexp_name[3]} -parent_dir $parent_dir -seed 45678 > /dev/null 2>&1 &
