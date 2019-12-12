exp_name="op_method_2"
run_file="/home/qzhou/aaai2019/POMBU/run.py"
subexp_name=("vc_kl_0" "vc_kl_0dot25" "p_improve_kl_0dot5" "p_improve_kl_1")
parent_dir="/home/qzhou/aaai_data"

rm -rf $parent_dir/$exp_name

CUDA_VISIBLE_DEVICES=0 python $run_file -env_name half_cheetah -exp_name $exp_name -subexp_name ${subexp_name[0]} -parent_dir $parent_dir -seed 12345 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=0 python $run_file -env_name half_cheetah -exp_name $exp_name -subexp_name ${subexp_name[0]} -parent_dir $parent_dir -seed 23456 > /dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=1 python $run_file -env_name half_cheetah -exp_name $exp_name -subexp_name ${subexp_name[0]} -parent_dir $parent_dir -seed 34567 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=1 python $run_file -env_name half_cheetah -exp_name $exp_name -subexp_name ${subexp_name[0]} -parent_dir $parent_dir -seed 45678 > /dev/null 2>&1 &


CUDA_VISIBLE_DEVICES=2 python $run_file -env_name half_cheetah -exp_name $exp_name -subexp_name ${subexp_name[1]} -parent_dir $parent_dir -seed 12345 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=2 python $run_file -env_name half_cheetah -exp_name $exp_name -subexp_name ${subexp_name[1]} -parent_dir $parent_dir -seed 23456 > /dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=3 python $run_file -env_name half_cheetah -exp_name $exp_name -subexp_name ${subexp_name[1]} -parent_dir $parent_dir -seed 34567 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=3 python $run_file -env_name half_cheetah -exp_name $exp_name -subexp_name ${subexp_name[1]} -parent_dir $parent_dir -seed 45678 > /dev/null 2>&1 &

#CUDA_VISIBLE_DEVICES=4 python $run_file -env_name half_cheetah -exp_name $exp_name -subexp_name ${subexp_name[2]} -parent_dir $parent_dir -seed 12345 > /dev/null 2>&1 &
#CUDA_VISIBLE_DEVICES=4 python $run_file -env_name half_cheetah -exp_name $exp_name -subexp_name ${subexp_name[2]} -parent_dir $parent_dir -seed 23456 > /dev/null 2>&1 &

#CUDA_VISIBLE_DEVICES=5 python $run_file -env_name half_cheetah -exp_name $exp_name -subexp_name ${subexp_name[2]} -parent_dir $parent_dir -seed 34567 > /dev/null 2>&1 &
#CUDA_VISIBLE_DEVICES=5 python $run_file -env_name half_cheetah -exp_name $exp_name -subexp_name ${subexp_name[2]} -parent_dir $parent_dir -seed 45678 > /dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=6 python $run_file -env_name half_cheetah -exp_name $exp_name -subexp_name ${subexp_name[3]} -parent_dir $parent_dir -seed 12345 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=6 python $run_file -env_name half_cheetah -exp_name $exp_name -subexp_name ${subexp_name[3]} -parent_dir $parent_dir -seed 23456 > /dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=7 python $run_file -env_name half_cheetah -exp_name $exp_name -subexp_name ${subexp_name[3]} -parent_dir $parent_dir -seed 34567 &
CUDA_VISIBLE_DEVICES=7 python $run_file -env_name half_cheetah -exp_name $exp_name -subexp_name ${subexp_name[3]} -parent_dir $parent_dir -seed 45678 > /dev/null 2>&1 &

