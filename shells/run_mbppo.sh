exp_name="mbppo"
env_name=("ant" "swimmer" "walker2d")
run_file="/home/qzhou/aaai2019/POMBU/run.py"
subexp_name=("vc_kl_0_ant" "vc_kl_0_swimmer" "vc_kl_0_walker2d")
parent_dir="/home/qzhou/aaai_data"

#rm -rf $parent_dir/$exp_name

#CUDA_VISIBLE_DEVICES=3 python $run_file -env_name ${env_name[0]} -exp_name $exp_name -subexp_name ${subexp_name[0]} -parent_dir $parent_dir -seed 12345 > /dev/null 2>&1 &
#CUDA_VISIBLE_DEVICES=3 python $run_file -env_name ${env_name[0]} -exp_name $exp_name -subexp_name ${subexp_name[0]} -parent_dir $parent_dir -seed 23456 > /dev/null 2>&1 &

#CUDA_VISIBLE_DEVICES=3 python $run_file -env_name ${env_name[0]} -exp_name $exp_name -subexp_name ${subexp_name[0]} -parent_dir $parent_dir -seed 34567 &
#CUDA_VISIBLE_DEVICES=3 python $run_file -env_name ${env_name[1]} -exp_name $exp_name -subexp_name ${subexp_name[1]} -parent_dir $parent_dir -seed 12345 > /dev/null 2>&1 &


#CUDA_VISIBLE_DEVICES=4 python $run_file -env_name ${env_name[1]} -exp_name $exp_name -subexp_name ${subexp_name[1]} -parent_dir $parent_dir -seed 23456 > /dev/null 2>&1 &
#CUDA_VISIBLE_DEVICES=4 python $run_file -env_name ${env_name[1]} -exp_name $exp_name -subexp_name ${subexp_name[1]} -parent_dir $parent_dir -seed 34567 > /dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=5 python $run_file -env_name ${env_name[2]} -exp_name $exp_name -subexp_name ${subexp_name[2]} -parent_dir $parent_dir -seed 12345 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=5 python $run_file -env_name ${env_name[2]} -exp_name $exp_name -subexp_name ${subexp_name[2]} -parent_dir $parent_dir -seed 23456 > /dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=5 python $run_file -env_name ${env_name[2]} -exp_name $exp_name -subexp_name ${subexp_name[2]} -parent_dir $parent_dir -seed 34567 > /dev/null 2>&1 &
#CUDA_VISIBLE_DEVICES=3 python $run_file -env_name ant -exp_name $exp_name -subexp_name ${subexp_name[2]} -parent_dir $parent_dir -seed 23456 > /dev/null 2>&1 &

#CUDA_VISIBLE_DEVICES=4 python $run_file -env_name ant -exp_name $exp_name -subexp_name ${subexp_name[2]} -parent_dir $parent_dir -seed 34567 > /dev/null 2>&1 &
#CUDA_VISIBLE_DEVICES=5 python $run_file -env_name ant -exp_name $exp_name -subexp_name ${subexp_name[2]} -parent_dir $parent_dir -seed 45678 > /dev/null 2>&1 &

#CUDA_VISIBLE_DEVICES=5 python $run_file -env_name ant -exp_name $exp_name -subexp_name ${subexp_name[3]} -parent_dir $parent_dir -seed 12345 > /dev/null 2>&1 &
#CUDA_VISIBLE_DEVICES=5 python $run_file -env_name ant -exp_name $exp_name -subexp_name ${subexp_name[3]} -parent_dir $parent_dir -seed 23456 > /dev/null 2>&1 &

#CUDA_VISIBLE_DEVICES=5 python $run_file -env_name ant -exp_name $exp_name -subexp_name ${subexp_name[3]} -parent_dir $parent_dir -seed 34567 > /dev/null 2>&1 &
#CUDA_VISIBLE_DEVICES=6 python $run_file -env_name ant -exp_name $exp_name -subexp_name ${subexp_name[3]} -parent_dir $parent_dir -seed 45678 > /dev/null 2>&1 &

#CUDA_VISIBLE_DEVICES=6 python $run_file -env_name ant -exp_name $exp_name -subexp_name ${subexp_name[4]} -parent_dir $parent_dir -seed 12345 > /dev/null 2>&1 &
#CUDA_VISIBLE_DEVICES=6 python $run_file -env_name ant -exp_name $exp_name -subexp_name ${subexp_name[4]} -parent_dir $parent_dir -seed 23456 > /dev/null 2>&1 &

#CUDA_VISIBLE_DEVICES=6 python $run_file -env_name ant -exp_name $exp_name -subexp_name ${subexp_name[4]} -parent_dir $parent_dir -seed 34567 > /dev/null 2>&1 &
#CUDA_VISIBLE_DEVICES=6 python $run_file -env_name ant -exp_name $exp_name -subexp_name ${subexp_name[3]} -parent_dir $parent_dir -seed 45678 > /dev/null 2>&1 &
