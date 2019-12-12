exp_name="op_method_2"
run_file="/home/qzhou/aaai2019/POMBU/run.py"
subexp_name=("vc_kl_0" "vc_kl_0dot25" "vc_kl_0dot5" "vc_kl_0dot75")
parent_dir="/home/qzhou/aaai_data"

rm -rf $parent_dir/$exp_name

python $run_file -env_name half_cheetah -exp_name $exp_name -subexp_name ${subexp_name[2]} -parent_dir $parent_dir
#CUDA_VISIBLE_DEVICES=0 python $run_file -env_name half_cheetah -exp_name $exp_name -subexp_name ${subexp_name[0]} -parent_dir $parent_dir 

#CUDA_VISIBLE_DEVICES=1 python $run_file -env_name half_cheetah -exp_name $exp_name -subexp_name ${subexp_name[0]} -parent_dir $parent_dir 
#CUDA_VISIBLE_DEVICES=1 python $run_file -env_name half_cheetah -exp_name $exp_name -subexp_name ${subexp_name[0]} -parent_dir $parent_dir 

#CUDA_VISIBLE_DEVICES=2 python $run_file -env_name half_cheetah -exp_name $exp_name -subexp_name ${subexp_name[1]} -parent_dir $parent_dir 
#CUDA_VISIBLE_DEVICES=2 python $run_file -env_name half_cheetah -exp_name $exp_name -subexp_name ${subexp_name[1]} -parent_dir $parent_dir 

#CUDA_VISIBLE_DEVICES=3 python $run_file -env_name half_cheetah -exp_name $exp_name -subexp_name ${subexp_name[1]} -parent_dir $parent_dir 
#CUDA_VISIBLE_DEVICES=3 python $run_file -env_name half_cheetah -exp_name $exp_name -subexp_name ${subexp_name[1]} -parent_dir $parent_dir 

#CUDA_VISIBLE_DEVICES=6 python $run_file -env_name half_cheetah -exp_name $exp_name -subexp_name ${subexp_name[3]} -parent_dir $parent_dir 
#CUDA_VISIBLE_DEVICES=6 python $run_file -env_name half_cheetah -exp_name $exp_name -subexp_name ${subexp_name[3]} -parent_dir $parent_dir 

#CUDA_VISIBLE_DEVICES=7 python $run_file -env_name half_cheetah -exp_name $exp_name -subexp_name ${subexp_name[3]} -parent_dir $parent_dir 
#CUDA_VISIBLE_DEVICES=7 python $run_file -env_name half_cheetah -exp_name $exp_name -subexp_name ${subexp_name[3]} -parent_dir $parent_dir 

