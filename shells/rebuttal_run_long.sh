exp_name="cheetah"
env_name="half_cheetah"
run_file="/home/qzhou/aaai2019/POMBU/run.py"
subexp_name=("vc_kl_0dot5")
parent_dir="/home/qzhou/aaai_data"


python $run_file -env_name $env_name -exp_name $exp_name -subexp_name ${subexp_name[1]} -parent_dir $parent_dir > /dev/null 2>&1 &
