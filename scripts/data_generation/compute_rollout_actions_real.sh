#!/bin/bash
set -e


DEBUG=1
SLURM_HOSTNAME="<enter_hostname>"
SLURM_SBATCH_FILE="<enter_sbatch_file>"


function run_cmd {
    echo ""
    echo ${CMD}
    if [[ ${DEBUG} == 0 ]]; then
        if [[ `hostname` == "${SLURM_HOSTNAME}" ]]; then
            sbatch "${SLURM_SBATCH_FILE}" "${CMD}"
        else
            eval ${CMD}
        fi
    fi
}


function add_save_cmd {
    # Encoder embeddings.
    if [[ ${save_encoder_embeddings} == 1 ]]; then
        CMD="${CMD} +eval.save_encoder_embeddings=1"
    fi
    # ResNet embeddings.
    if [[ ${save_resnet_embeddings} == 1 ]]; then
        CMD="${CMD} +eval.save_resnet_embeddings=1"
    fi
    # CLIP embeddings.
    if [[ ${save_clip_embeddings} == 1 ]]; then
        CMD="${CMD} +eval.save_clip_embeddings=1"
    fi

    CMD="${CMD} +eval.embedding_model_device=${embedding_model_device}" 

    if [[ ${save_rec_actions} == 1 ]]; then
        if [[ "${agent}" == *"sim3"* ]]; then
            echo "Error: save_rec_actions is not supported for SIM(3) agents."
            exit 1
        fi
        CMD="${CMD} +eval.save_rec_actions=1"
        local rec_depths_str="${rec_depths[*]}"
        rec_depths_str="${rec_depths_str// /,}"
        CMD="${CMD} +model.rec_depths=[${rec_depths_str}]"
    fi

    if [[ ${save_noise_preds} == 1 ]]; then
        if [[ "${agent}" == *"sim3"* ]]; then
            echo "Error: save_noise_preds is not supported for SIM(3) agents."
            exit 1
        fi
        CMD="${CMD} +eval.save_noise_preds=1"
        CMD="${CMD} +model.noise_pred_samples=${noise_pred_samples}"
    fi
}


function compute_rollout_actions_real {
    ckpt_path="${ckpt_paths[${domain}_${agent}]}"
    ckpt_name="${ckpt_names[${domain}_${agent}]}"
    max_episode_length="${max_episode_lengths[${domain}_${agent}]}"

    # Construct command.
    CMD="python -m sentinel.bc.${script} --config-name ${robot_setup}_${agent} robot_info=${domain} mode=eval seed=${seed}"
    CMD="${CMD} training.ckpt=${ckpt_path}/${ckpt_name}.pth training.batch_size=${batch_size} robot_info.use_dummy_zed=true robot_info.freq=${robot_freq}"
    CMD="${CMD} env.args.max_episode_length=${max_episode_length} env.args.ac_scale=${action_scale} model.ac_horizon=${exec_horizon}"
    CMD="${CMD} +eval.save_episodes=${save_episodes} +eval.save_videos=${save_videos} +eval.overwrite=${overwrite}"
    CMD="${CMD} log_dir=${log_dir} prefix=${date}_${domain}_${agent}_${tag} use_wandb=false"
    add_save_cmd
    run_cmd
}


# Setup.
date="<enter_date>"
script="compute_rollout_actions_real"
log_dir="logs/bc/real_eval"

# Paths.
train_dir="${PWD}/logs/bc/train"
declare -A ckpt_paths=(
    #### Official experiment checkpoint. ####

    ## SIM(3) diffusion policy: Trained on real-world demonstrations.
    ["push_chair_sim3_dp_ddim"]="${train_dir}/push_chair/0603_push_chair_3hz_sim3_dp_ep50"

    #### Unofficial checkpoints, included for completion and prototyping. ####

    # Diffusion policy: Trained on real-world demonstrations.
    # ["push_chair_dp_ddim"]="${train_dir}/push_chair/0603_push_chair_3hz_dp_ep50"
)
declare -A ckpt_names=(
    #### Official experiment checkpoints. ####

    ## SIM(3) diffusion policy: Trained on real-world demonstrations.
    ["push_chair_sim3_dp_ddim"]="ckpt00999"

    #### Unofficial checkpoints, included for completion and prototyping. ####

    ## Diffusion policy: Trained on real-world demonstrations.
    # ["push_chair_dp_ddim"]="ckpt00999"
)

# Episodes.
declare -A max_episode_lengths=(
    ## Diffusion policy.
    ["push_chair_dp_ddim"]=100
    ## SIM(3) diffusion policy.
    ["push_chair_sim3_dp_ddim"]=100
)

# Embedding baseline data.
save_encoder_embeddings=1
save_resnet_embeddings=1
save_clip_embeddings=1
embedding_model_device="cuda"

# Empirical loss (noise prediction) baseline (not supported for SIM(3) agents, e.g., "sim3_dp_ddim" below).
save_noise_preds=0
noise_pred_samples=8

# Reconstruction baseline (not supported for SIM(3) agents, e.g., "sim3_dp_ddim" below).
save_rec_actions=0
rec_depths=(
    1
    2
    4
    8
)

### Task: Push Chair | SIM(3) diffusion policy.
domain="push_chair"
robot_setup="real_single_arm"
agent="sim3_dp_ddim"    # SIM(3) diffusion policy; official agent.
# agent="dp_ddim"       # Diffusion policy; unofficial agent.

# Real domain parameters.
seed=0
save_episodes=1
save_videos=1
overwrite=0
batch_size=256
exec_horizon=4
robot_freq=3
action_scale=3

# Calibration set: success.
tag="calib"
compute_rollout_actions_real

# Test set: success / failure.
tag="test"
compute_rollout_actions_real