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
    CMD="python -m sentinel.bc.${script} --config-name ${domain}_mobile_${agent} mode=eval seed=${seed} use_wandb=false"
    CMD="${CMD} training.ckpt=${ckpt_path}/${ckpt_name}.pth training.batch_size=${batch_size}"
    CMD="${CMD} env.args.ac_scale=${action_scale} model.ac_horizon=${exec_horizon}"
    CMD="${CMD} env.args.scale_mode=${scale_mode} env.args.deform_randomize_scale=${randomize_scale}"
    CMD="${CMD} env.args.max_episode_length=${max_episode_length} env.args.freq=${sim_freq} +eval.reward_thresh=${reward_thresh}"
    CMD="${CMD} +eval.num_eval_episodes=${num_eval_episodes} +eval.save_episodes=${save_episodes} +eval.save_videos=${save_videos}"
    CMD="${CMD} env.args.cam_resolution=${cam_resolution}"
}


declare -A random_rigid_object_positions=(
    ## Cover domain randomizations. ##
    # CA: In-distribution calibration split; Negligible domain randomization.
    ["cover_ca_rigid_pos_x_low"]=-0.1
    ["cover_ca_rigid_pos_x_high"]=0.1
    ["cover_ca_rigid_pos_y_low"]=0.0
    ["cover_ca_rigid_pos_y_high"]=0.1
    # NA: In-distribution test split; Negligible domain randomization.
    ["cover_na_rigid_pos_x_low"]=-0.1
    ["cover_na_rigid_pos_x_high"]=0.1
    ["cover_na_rigid_pos_y_low"]=0.0
    ["cover_na_rigid_pos_y_high"]=0.1
    # LL: Out-of-distribution test split; Light domain randomization.
    ["cover_ll_rigid_pos_x_low"]=-0.15
    ["cover_ll_rigid_pos_x_high"]=0.15
    ["cover_ll_rigid_pos_y_low"]=0.0
    ["cover_ll_rigid_pos_y_high"]=0.15
    # HH: Out-of-distribution test split; Heavy domain randomization (erratic failures).
    ["cover_hh_rigid_pos_x_low"]=-0.25
    ["cover_hh_rigid_pos_x_high"]=0.25
    ["cover_hh_rigid_pos_y_low"]=0.0
    ["cover_hh_rigid_pos_y_high"]=0.25
    # SS: Out-of-distribution test split; Smooth policy behavior (task progression failures).
    ["cover_ss_rigid_pos_x_low"]=-0.25
    ["cover_ss_rigid_pos_x_high"]=0.25
    ["cover_ss_rigid_pos_y_low"]=0.25
    ["cover_ss_rigid_pos_y_high"]=0.30
)
function add_random_rigid_object_positions_cmd {
    CMD="${CMD} +env.args.randomize_rigid_pos=1"
    
    for p in "rigid_pos_x_low" "rigid_pos_x_high" "rigid_pos_y_low" "rigid_pos_y_high"; do
        CMD="${CMD} +env.args.${p}=${random_rigid_object_positions[${domain}_${randomization}_${p}]}"
    done
}


declare -A random_rigid_object_scales=(
    ## Close domain randomizations. ##
    # CA: In-distribution calibration split; Negligible domain randomization.
    ["close_ca_real_src_scale_low"]=1.00
    ["close_ca_real_src_scale_high"]=1.30
    # NA: In-distribution test split; Negligible domain randomization.
    ["close_na_real_src_scale_low"]=1.00
    ["close_na_real_src_scale_high"]=1.30
    # LL: Out-of-distribution test split; Light domain randomization.
    ["close_ll_real_src_scale_low"]=1.40
    ["close_ll_real_src_scale_high"]=1.75
    # HH: Out-of-distribution test split; Heavy domain randomization (erratic failures).
    ["close_hh_real_src_scale_low"]=1.65
    ["close_hh_real_src_scale_high"]=1.90
    # SS: Out-of-distribution test split; Smooth policy behavior (task progression failures).
    ["close_ss_real_src_scale_low"]=1.00
    ["close_ss_real_src_scale_high"]=2.30
)
function add_random_rigid_object_scales_cmd {
    CMD="${CMD} +env.args.randomize_rigid_scale=1 +env.args.randomize_soft_scale=0"

    for p in "scale_low" "scale_high"; do
        CMD="${CMD} env.args.${p}=${random_rigid_object_scales[${domain}_${randomization}_${scale_mode}_${p}]}"
    done
}


declare -A random_soft_dynamics=(
    ## Fold domain randomizations (unused domain; set parameters as desired). ##
    # CA: In-distribution calibration split; Negligible domain randomization.
    ["fold_ca_deform_mass_percent"]=TBD.
    ["fold_ca_deform_bending_stiffness_percent"]=TBD.
    ["fold_ca_deform_damping_stiffness_percent"]=-TBD.
    ["fold_ca_deform_elastic_stiffness_percent"]=TBD.
    ["fold_ca_deform_friction_coeff_percent"]=TBD.
    # NA: In-distribution test split; Negligible domain randomization.
    ["fold_na_deform_mass_percent"]=TBD.
    ["fold_na_deform_bending_stiffness_percent"]=TBD.
    ["fold_na_deform_damping_stiffness_percent"]=-TBD.
    ["fold_na_deform_elastic_stiffness_percent"]=TBD.
    ["fold_na_deform_friction_coeff_percent"]=TBD.
    # LL: Out-of-distribution test split; Light domain randomization.
    ["fold_ll_deform_mass_percent"]=TBD.
    ["fold_ll_deform_bending_stiffness_percent"]=TBD.
    ["fold_ll_deform_damping_stiffness_percent"]=-TBD.
    ["fold_ll_deform_elastic_stiffness_percent"]=TBD.
    ["fold_ll_deform_friction_coeff_percent"]=TBD.
    # HH: Out-of-distribution test split; Heavy domain randomization (erratic failures).
    ["fold_hh_deform_mass_percent"]=TBD.
    ["fold_hh_deform_bending_stiffness_percent"]=TBD.
    ["fold_hh_deform_damping_stiffness_percent"]=-TBD.
    ["fold_hh_deform_elastic_stiffness_percent"]=TBD.
    ["fold_hh_deform_friction_coeff_percent"]=TBD.
)
function add_random_soft_dynamics_cmd {
    CMD="${CMD} +env.args.randomize_soft_dynamics=1"

    for p in "deform_mass_percent" "deform_bending_stiffness_percent" "deform_damping_stiffness_percent" "deform_elastic_stiffness_percent" "deform_friction_coeff_percent"; do
        CMD="${CMD} +env.args.${p}=${random_soft_dynamics[${domain}_${randomization}_${p}]}"
    done
}


declare -A random_eef_robot_positions=(
    ## Fold domain randomizations. ##
    ["fold_ca_robot_eef_max_offset"]=0.025
    ["fold_na_robot_eef_max_offset"]=0.025
    ["fold_ll_robot_eef_max_offset"]=0.025
    ["fold_hh_robot_eef_max_offset"]=0.025
    ## Cover domain randomizations. ##
    ["cover_ca_robot_eef_max_offset"]=0.025
    ["cover_na_robot_eef_max_offset"]=0.025
    ["cover_ll_robot_eef_max_offset"]=0.025
    ["cover_hh_robot_eef_max_offset"]=0.025
    ["cover_ss_robot_eef_max_offset"]=0.025
    ## Close domain randomizations. ##
    ["close_ca_robot_eef_max_offset"]=0.025
    ["close_na_robot_eef_max_offset"]=0.025
    ["close_ll_robot_eef_max_offset"]=0.025
    ["close_hh_robot_eef_max_offset"]=0.025
    ["close_ss_robot_eef_max_offset"]=0.100
)
function add_random_eef_robot_positions_cmd {
    CMD="${CMD} +env.args.randomize_robot_eef_pos=1"

    for p in "robot_eef_max_offset"; do
        CMD="${CMD} +env.args.${p}=${random_eef_robot_positions[${domain}_${randomization}_${p}]}"
    done

    if [[ "${randomization}" != "na" ]]; then
        CMD="${CMD} +env.args.grasp_threshold=${grasp_threshold}"
    fi
}


declare -A random_dynamics_noise=(
    ## Fold domain randomizations. ##
    ["fold_ll_mean"]=0.0
    ["fold_ll_std_dev"]=0.02
    ["fold_hh_mean"]=0.0
    ["fold_hh_std_dev"]=0.05
    ## Cover domain randomizations. ##
    ["cover_ll_mean"]=0.0
    ["cover_ll_std_dev"]=0.05
    ["cover_hh_mean"]=0.0
    ["cover_hh_std_dev"]=0.075
    ## Close domain randomizations. ##
    ["close_ll_mean"]=0.0
    ["close_ll_std_dev"]=0.05
    ["close_hh_mean"]=0.0
    ["close_hh_std_dev"]=0.075
)
function add_random_dynamics_noise_cmd {
    CMD="${CMD} +env.args.randomize_dynamics_noise=1"
    CMD="${CMD} +env.args.dynamics_noise_scheduler=GaussianNoiseScheduler"
    
    for p in "mean" "std_dev"; do
        CMD="${CMD} +env.args.dynamics_noise_scheduler_kwargs.${p}=${random_dynamics_noise[${domain}_${randomization}_${p}]}"
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
            if [[ "${randomize_rigid_pos}" == 1 ]]; then
                if [[ "${domain}" != "cover" ]]; then
                    echo "Domain ${domain} does not support rigid object position randomization."
                    exit 1
                fi
                add_random_rigid_object_positions_cmd
            fi
            prefix="${prefix}_rigid_pos_${randomize_rigid_pos}"

            # Randomize rigid object scales.
            if [[ "${randomize_rigid_scale}" == 1 ]]; then
                if [[ "${domain}" != "close" ]]; then
                    echo "Domain ${domain} does not support rigid object scale randomization."
                    exit 1
                fi
                add_random_rigid_object_scales_cmd   
            fi
            prefix="${prefix}_rigid_scale_${randomize_rigid_scale}"

            # Randomize soft body dynamics.
            if [[ "${randomize_soft_dynamics}" == 1 ]]; then
                if [[ "${domain}" != "fold" ]]; then
                    echo "Domain ${domain} does not support soft dynamics randomization."
                    exit 1
                fi
                add_random_soft_dynamics_cmd
            fi
            prefix="${prefix}_soft_dynamics_${randomize_soft_dynamics}"

            # Randomize robot end-effector positions.
            if [[ "${randomize_robot_eef_pos}" == 1 ]]; then
                add_random_eef_robot_positions_cmd
            fi
            prefix="${prefix}_robot_eef_pos_${randomize_robot_eef_pos}"

            # Randomize environment dynamics (Gaussian noise).
            if [[ "${randomize_dynamics_noise}" == 1 ]]; then
                add_random_dynamics_noise_cmd
            fi
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
    "dp"
    ## SIM(3) diffusion policy.
    # "sim3_dp"
)

# Paths.
train_dir="${PWD}/logs/bc/train"
declare -A ckpt_paths=(
    #### Official experiment checkpoints. ####

    ## Diffusion policy: Trained with light domain randomization, multimodal demos, and no dynamics noise.
    ["cover_sim_dp"]="${train_dir}/cover/0511_sim_mobile_n0_cover_7dof_dp_s0"
    ["close_sim_dp"]="${train_dir}/close/0511_sim_mobile_n0_close_7dof_dp_s0"
    
    #### Unofficial checkpoints, included for completion and prototyping. ####
    
    ## Diffusion policy: Trained with light domain randomization, multimodal demos, and light dynamics noise.
    # ["cover_sim_dp"]="${train_dir}/cover/0511_sim_mobile_n0.05_cover_7dof_dp_s0"
    # ["close_sim_dp"]="${train_dir}/close/0511_sim_mobile_n0.05_close_7dof_dp_s0"

    ## Diffusion policy: Trained with no domain randomization, and unimodal demos.
    # ["fold_sim_dp"]="${train_dir}/fold/0429_sim_mobile_fold_7dof_dp_s1"
    # ["cover_sim_dp"]="${train_dir}/cover/0429_sim_mobile_cover_7dof_dp_s1"
    # ["close_sim_dp"]="${train_dir}/close/0429_sim_mobile_close_7dof_dp_s1"

    ## SIM(3) diffusion policy: Trained with no domain randomization, and unimodal demos.
    # ["fold_sim_sim3_dp"]="${train_dir}/fold/0429_sim_mobile_fold_7dof_sim3_dp_s1"
    # ["cover_sim_sim3_dp"]="${train_dir}/cover/0429_sim_mobile_cover_7dof_sim3_dp_s1"
    # ["close_sim_sim3_dp"]="${train_dir}/close/0429_sim_mobile_close_7dof_sim3_dp_s1"

)
declare -A ckpt_names=(
    #### Official experiment checkpoints. ####
    
    ## Diffusion policy: Trained with light domain randomization, multimodal demos, and no dynamics noise.
    ["cover_sim_dp"]="ckpt00999"
    ["close_sim_dp"]="ckpt00999"
    
    #### Unofficial checkpoints, included for completion and prototyping. ####
    
    ## Diffusion policy: Trained with light domain randomization, multimodal demos, and light dynamics noise.
    # ["cover_sim_dp"]="ckpt00999"
    # ["close_sim_dp"]="ckpt00999"

    ## Diffusion policy: Trained with no domain randomization, and unimodal demos.
    # ["fold_sim_dp"]="ckpt01999"
    # ["cover_sim_dp"]="ckpt01999"
    # ["close_sim_dp"]="ckpt01999"
    
    ## SIM(3) diffusion policy: Trained with no domain randomization, and unimodal demos.
    # ["fold_sim_sim3_dp"]="ckpt01999"
    # ["cover_sim_sim3_dp"]="ckpt01999"
    # ["close_sim_sim3_dp"]="ckpt01999"
)

# Reward thresholds.
declare -A reward_thresholds=(
    ## Diffusion policy.
    ["fold_sim_dp"]=0.5
    ["cover_sim_dp"]=0.75
    ["close_sim_dp"]=0.75
    ## SIM(3) diffusion policy.
    ["fold_sim_sim3_dp"]=0.5
    ["cover_sim_sim3_dp"]=0.75
    ["close_sim_sim3_dp"]=0.75
)

# Episodes.
declare -A max_episode_lengths=(
    ## Diffusion policy.
    ["fold_sim_dp"]=60
    ["cover_sim_dp"]=80
    ["close_sim_dp"]=120
    ## SIM(3) diffusion policy.
    ["fold_sim_sim3_dp"]=60
    ["cover_sim_sim3_dp"]=80
    ["close_sim_sim3_dp"]=120
)

# Experiment parameters.
seed=0
num_eval_episodes=50
save_episodes=1
save_videos=1
cam_resolution=512

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

# Mobile domain parameters.
demos="sim"
sim_freq=5
scale_mode="real_src"
randomize_scale=1
batch_size=256
action_scale=1
grasp_threshold=0.04
exec_horizons=(
    # 2
    4 
    # 8
)


#### Official experiment domains. ####

## Experiment: Cover.
domain="cover"
randomizations=(
    # "ca"
    # "na"
    # "ll"
    # "hh"
    # "ss"
)
for randomization in "${randomizations[@]}"; do   
    randomize_rigid_pos=1
    randomize_rigid_scale=0
    randomize_robot_eef_pos=1
    randomize_soft_dynamics=0
    randomize_dynamics_noise=0
    # compute_rollout_actions
done

## Experiment: Close.
domain="close"
randomizations=(
    # "ca"
    # "na"
    # "ll"
    # "hh"
    # "ss"
)
for randomization in "${randomizations[@]}"; do   
    randomize_rigid_pos=0
    randomize_rigid_scale=1
    randomize_robot_eef_pos=1
    randomize_soft_dynamics=0
    randomize_dynamics_noise=0
    # compute_rollout_actions
done


#### Unofficial domain, included for completion and prototyping. ####

## Experiment: Fold.
domain="fold"
randomizations=(
    # "ca"
    # "na"
    # "ll"
    # "hh"
)
for randomization in "${randomizations[@]}"; do   
    randomize_rigid_pos=0
    randomize_rigid_scale=0
    randomize_robot_eef_pos=1
    randomize_soft_dynamics=1
    randomize_dynamics_noise=0
    # compute_rollout_actions
done