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


function get_base_cmd {
    CMD="python -m sentinel.bc.${script} --config-name ${domain}_synthetic_${agent} mode=eval seed=${seed}"
    CMD="${CMD} training.ckpt=${ckpt_path}/${ckpt_name}.pth training.batch_size=${batch_size}"
    CMD="${CMD} env.args.ac_scale=${action_scale} model.ac_horizon=${exec_horizon} model.obs_mode=pc2 model.ac_mode=abs"
    CMD="${CMD} env.args.scale_mode=${scale_mode} env.args.deform_randomize_scale=${randomize_scale} env.args.randomize_rotation=${randomize_rotation}"
    CMD="${CMD} env.args.max_episode_length=${max_episode_length} +env.args.term_on_success=${term_on_thresh} +eval.reward_thresh=${reward_thresh} +eval.term_on_thresh=${term_on_thresh}"
    CMD="${CMD} +eval.num_eval_episodes=${num_eval_episodes} +eval.save_episodes=${save_episodes} +eval.save_videos=${save_videos}"
    CMD="${CMD} data.dataset_class=cloth_synthetic +env.args.render_action=false +eval.overwrite=${overwrite} use_wandb=false"
}


declare -A random_rigid_object_scales=(
    ## Pusht domain randomizations. ##
    # CA: In-distribution calibration split; Negligible domain randomization.
    ["pusht_ca_real_src_scale_low"]=1.00
    ["pusht_ca_real_src_scale_high"]=1.30
    ["pusht_ca_real_src_uniform_scaling"]=1
    # NA: In-distribution test split; Negligible domain randomization.
    ["pusht_na_real_src_scale_low"]=1.00
    ["pusht_na_real_src_scale_high"]=1.30
    ["pusht_na_real_src_uniform_scaling"]=1
    # LL: Out-of-distribution test split; Light domain randomization.
    ["pusht_ll_real_src_scale_low"]=1.00
    ["pusht_ll_real_src_scale_high"]=1.75
    ["pusht_ll_real_src_uniform_scaling"]=1
    # HH: Out-of-distribution test split; Heavy domain randomization (erratic failures).
    ["pusht_hh_real_src_scale_low"]=1.00
    ["pusht_hh_real_src_scale_high"]=2.00
    ["pusht_hh_real_src_scale_aspect_limit"]=1.33
    ["pusht_hh_real_src_uniform_scaling"]=0
    # SS: Out-of-distribution test split; Smooth policy behavior (task progression failures).
    ["pusht_ss_real_src_scale_low"]=1.00
    ["pusht_ss_real_src_scale_high"]=2.00
    ["pusht_ss_real_src_scale_aspect_limit"]=1.30
)
function add_random_rigid_object_scales_cmd {
    CMD="${CMD} +env.args.randomize_rigid_scale=1 +env.args.randomize_soft_scale=0"

    for p in "scale_low" "scale_high" "scale_aspect_limit" "uniform_scaling"; do
        if [[ -n "${random_rigid_object_scales[${domain}_${randomization}_${scale_mode}_${p}]}" ]]; then
            CMD="${CMD} env.args.${p}=${random_rigid_object_scales[${domain}_${randomization}_${scale_mode}_${p}]}"
        fi 
    done
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
        CMD="${CMD} +eval.save_rec_actions=1"
        local rec_depths_str="${rec_depths[*]}"
        rec_depths_str="${rec_depths_str// /,}"
        CMD="${CMD} +model.rec_depths=[${rec_depths_str}]"
    fi

    if [[ ${save_noise_preds} == 1 ]]; then
        CMD="${CMD} +eval.save_noise_preds=1"
        CMD="${CMD} +model.noise_pred_samples=${noise_pred_samples}"
    fi

    if [[ ${save_score_pairs} == 1 ]]; then
        CMD="${CMD} +eval.save_score_pairs=1"
    fi
}


function compute_rollout_actions {
    for agent in "${agents[@]}"; do
        for exec_horizon in "${exec_horizons[@]}"; do
            ckpt_path="${ckpt_paths[${domain}_${demos}_${agent}]}"
            ckpt_name="${ckpt_names[${domain}_${demos}_${agent}]}"
            max_episode_length="${max_episode_lengths[${domain}_${demos}_${agent}]}"
            reward_thresh="${reward_thresholds[${domain}_${demos}_${agent}]}"
            get_base_cmd
            add_save_cmd

            if [[ "${randomize_rigid_pos}" == 1 || "${randomize_rigid_scale}" == 1 || "${randomize_soft_dynamics}" == 1 || "${randomize_robot_eef_pos}" == 1 || "${randomize_dynamics_noise}" == 1 ]]; then
                case "${randomization}" in
                    ca|na|ll|hh|ss)
                        ;;
                    *)
                        echo "The 'randomization' parameter is not set correctly."
                        exit 1
                        ;;
                esac
            fi

            prefix="${date}_${demos}_${script}_${domain}_${agent}_${ckpt_name}_exec_horizon_${exec_horizon}_randomization_${randomization}"

            # Randomize rigid object positions.
            prefix="${prefix}_rigid_pos_${randomize_rigid_pos}"

            # Randomize rigid object scales.
            if [[ "${randomize_rigid_scale}" == 1 ]]; then
                if [[ "${domain}" != "close" && "${domain}" != "pusht" ]]; then
                    echo "Domain ${domain} does not support rigid object scale randomization."
                    exit 1
                fi
                add_random_rigid_object_scales_cmd   
            fi
            prefix="${prefix}_rigid_scale_${randomize_rigid_scale}"

            # Randomize soft body dynamics.
            prefix="${prefix}_soft_dynamics_${randomize_soft_dynamics}"

            # Randomize robot end-effector positions.
            prefix="${prefix}_robot_eef_pos_${randomize_robot_eef_pos}"

            # Randomize environment dynamics (Gaussian noise).
            prefix="${prefix}_dynamics_noise_${randomize_dynamics_noise}"
            
            CMD="${CMD} prefix=${prefix}"
            run_cmd
        done
    done
}


# Setup.
date="<enter_date>"
script="compute_rollout_actions"

# Agents.
agents=(
    ## Diffusion policy.
    # "dp"
    ## Diffusion policy with data augmentation.
    "dp_aug"
    ## SIM(3) diffusion policy.
    # "sim3_dp"
)

# Paths.
train_dir="${PWD}/logs/bc/train"
declare -A ckpt_paths=(
    #### Official experiment checkpoints. ####

    ## Diffusion policy: Trained with light data augmentation.
    ["pusht_sim_dp_aug"]="${train_dir}/pusht/0503_sim_pusht_3dof_dp_aug_s1"    

    #### Unofficial checkpoints, included for completion and prototyping. ####

    ## Diffusion policy: Trained with negligible data augmentation.
    # ["pusht_sim_dp"]="${train_dir}/pusht/0503_sim_pusht_3dof_dp_s1"
    
    ## SIM(3) diffusion policy: Trained with negligible data augmentation.
    # ["pusht_sim_sim3_dp"]="${train_dir}/pusht/0503_sim_pusht_3dof_sim3_dp_s1"
)
declare -A ckpt_names=(
    #### Official experiment checkpoints. ####
    
    ## Diffusion policy: Trained with light data augmentation.
    ["pusht_sim_dp_aug"]="ckpt01999"    
    
    #### Unofficial checkpoints, included for completion and prototyping. ####

    ## Diffusion policy: Trained with negligible data augmentation.
    # ["pusht_sim_dp"]="ckpt01999"
    
    ## SIM(3) diffusion policy: Trained with negligible data augmentation.
    # ["pusht_sim_sim3_dp"]="ckpt01999"
)

# Reward thresholds.
declare -A reward_thresholds=(
    ## Diffusion policy.
    ["pusht_sim_dp"]=0.90
    ## Diffusion policy with data augmentation.
    ["pusht_sim_dp_aug"]=0.90
    ## SIM(3) diffusion policy.
    ["pusht_sim_sim3_dp"]=0.90
)

# Episodes.
declare -A max_episode_lengths=(
    ## Diffusion policy.
    ["pusht_sim_dp"]=300
    ## Diffusion policy with data augmentation.
    ["pusht_sim_dp_aug"]=300
    ## SIM(3) diffusion policy.
    ["pusht_sim_sim3_dp"]=300
)

# Experiment parameters.
seed=0
num_eval_episodes=50
term_on_thresh=1
save_episodes=1
save_videos=1
overwrite=0

# Embedding baseline data.
save_encoder_embeddings=1
save_resnet_embeddings=1
save_clip_embeddings=1
embedding_model_device="cuda"

# Empirical loss (noise prediction) baseline.
save_noise_preds=1
noise_pred_samples=10

# Score matching baseline.
save_score_pairs=0

# Reconstruction baseline.
save_rec_actions=1
rec_depths=(
    5
    10
    25
    50
    # 100
)

# Synthetic domain parameters.
demos="sim"
scale_mode="real_src"
randomize_scale=1
randomize_rotation=0
batch_size=256
action_scale=1
exec_horizons=(
    # 2
    # 4
    8
)

## Experiment: PushT.
domain="pusht"
randomizations=(
    # "ca"
    # "na"
    # "ll"
    # "hh"
)
for randomization in "${randomizations[@]}"; do   
    randomize_rigid_pos=0
    randomize_rigid_scale=1
    randomize_robot_eef_pos=0
    randomize_soft_dynamics=0
    randomize_dynamics_noise=0
    # compute_rollout_actions
done
