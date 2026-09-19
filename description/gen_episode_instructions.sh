task_name=${1}
setting=${2}
max_num=${3}
config_name=${4}

python utils/generate_episode_instructions.py $task_name $setting $max_num $config_name
