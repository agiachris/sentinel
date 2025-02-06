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
    CMD="${CMD} env.args.ac_scale=${action_scale} model.ac_horizon=${exec_horizon}"
    CMD="${CMD} env.args.scale_mode=${scale_mode} env.args.deform_randomize_scale=${randomize_scale} env.args.randomize_rotation=${randomize_rotation}"
    CMD="${CMD} env.args.max_episode_length=${max_episode_length} +eval.max_num_episodes=${max_num_episodes}"
    CMD="${CMD} +env.args.freq=${sim_freq} data.dataset_class=episode_dataset"
}


## Method: Statistical temporal action consistency (STAC).
# Note: Configuration below uses official hyperparameters.
function add_temporal_consistency_cmd {
    local pred_horizons=(
        # 8
        # 12
        16
    )
    local sample_sizes=(
        # 4
        # 8 
        # 16 
        # 32 
        # 64 
        # 128
        256
        # 512
    )
    local error_fns=(
        # Min.
        "mse_all"
        # "mse_pos"
        # "ate_pos"
        # MMD.
        # "mmd_rbf_all"
        "mmd_rbf_all_median"
        # "mmd_rbf_all_eig"
        # "mmd_rbf_all_0.1"
        # "mmd_rbf_all_0.5"
        # "mmd_rbf_all_1.0"
        # "mmd_rbf_all_5.0"
        # "mmd_rbf_all_10.0" 
        # "mmd_rbf_all_100.0" 
        # KDE For. KL.
        # "kde_kl_all_for"
        "kde_kl_all_for_eig"
        # "kde_kl_all_for_0.1"
        # "kde_kl_all_for_0.5"
        # "kde_kl_all_for_5.0"
        # "kde_kl_all_for_10.0"
        # "kde_kl_all_for_100.0"
        # KDE Rev. KL.
        # "kde_kl_all_rev"
        "kde_kl_all_rev_eig"
        # "kde_kl_all_rev_0.1"
        # "kde_kl_all_rev_0.5"
        # "kde_kl_all_rev_5.0"
        # "kde_kl_all_rev_10.0"
        # "kde_kl_all_rev_100.0"
    )
    local aggr_fns=(
        "min"
        # "max"
        # "mean"
    )

    local pred_horizons_str="${pred_horizons[*]}"
    local sample_sizes_str="${sample_sizes[*]}"
    local error_fns_str="${error_fns[*]}"
    local aggr_fns_str="${aggr_fns[*]}"

    pred_horizons_str="${pred_horizons_str// /,}"
    sample_sizes_str="${sample_sizes_str// /,}"
    error_fns_str="${error_fns_str// /,}"
    aggr_fns_str="${aggr_fns_str// /,}"
    
    CMD="${CMD} +eval.consistency.pred_horizons=[${pred_horizons_str}]"
    CMD="${CMD} +eval.consistency.sample_sizes=[${sample_sizes_str}]"
    CMD="${CMD} +eval.consistency.error_fns=[${error_fns_str}]"
    CMD="${CMD} +eval.consistency.aggr_fns=[${aggr_fns_str}]"
}


## Method: Diffusion output variance baseline.
# Note: Configuration below uses official hyperparameters.
function add_diffusion_ensemble_cmd {
    local pred_horizons=(
        # 12
        16
    )
    local sample_sizes=(
        # 4
        # 8 
        # 16 
        # 32 
        # 64 
        # 128
        256
        # 512
    )
    local action_spaces=(
        "all"
        # "pos"
        # "traj"
    )

    local pred_horizons_str="${pred_horizons[*]}"
    local sample_sizes_str="${sample_sizes[*]}"
    local action_spaces_str="${action_spaces[*]}"

    pred_horizons_str="${pred_horizons_str// /,}"
    sample_sizes_str="${sample_sizes_str// /,}"
    action_spaces_str="${action_spaces_str// /,}"
    
    CMD="${CMD} +eval.ensemble.pred_horizons=[${pred_horizons_str}]"
    CMD="${CMD} +eval.ensemble.sample_sizes=[${sample_sizes_str}]"
    CMD="${CMD} +eval.ensemble.action_spaces=[${action_spaces_str}]"
}


## Method: Embedding similarity baseline.
# Note: Configuration below uses official hyperparameters.
function add_embedding_similarity_cmd {
    local embeddings=(
        # "state_feat"
        "encoder_feat"
        "resnet_feat"
        "clip_feat"
    )
    local score_fns=(
        # "top1_l2"
        # "top5_l2"
        # "top10_l2"
        # "top1_cosine"
        # "top5_cosine"
        # "top10_cosine"
        "mahal"
    )

    local embeddings_str="${embeddings[*]}"
    local score_fns_str="${score_fns[*]}"

    embeddings_str="${embeddings_str// /,}"
    score_fns_str="${score_fns_str// /,}"

    CMD="${CMD} +eval.embedding.embeddings=[${embeddings_str}]"
    CMD="${CMD} +eval.embedding.score_fns=[${score_fns_str}]"
}


## Method: Likelihood / loss function baselines.
# Note: Configuration below uses official hyperparameters.
function add_loss_function_cmd {
    local loss_functions=(
        # "score_matching_all"
        # "score_matching_pos"
        "noise_pred_all"
        "temporal_noise_pred_all"
        "action_rec_all"
        "temporal_action_rec_all"
    )
    local score_matching_sample_sizes=(
        # 1
        # 5
        # 25
        # 50
        # 100
    )
    local noise_pred_sample_sizes=(
        # 1
        # 5
        10
        # 25
        # 50
        # 100
    )
    local action_rec_sample_sizes=(
        # 1
        # 2
        # 3
        4
        # 5
    )
    
    local loss_functions_str="${loss_functions[*]}"
    local score_matching_sample_sizes_str="${score_matching_sample_sizes[*]}"
    local noise_pred_sample_sizes_str="${noise_pred_sample_sizes[*]}"
    local action_rec_sample_sizes_str="${action_rec_sample_sizes[*]}"

    loss_functions_str="${loss_functions_str// /,}"
    score_matching_sample_sizes_str="${score_matching_sample_sizes_str// /,}"
    noise_pred_sample_sizes_str="${noise_pred_sample_sizes_str// /,}"
    action_rec_sample_sizes_str="${action_rec_sample_sizes_str// /,}"

    CMD="${CMD} +eval.loss_functions.loss_fns=[${loss_functions_str}]"
    CMD="${CMD} +eval.loss_functions.score_matching_sample_sizes=[${score_matching_sample_sizes_str}]"
    CMD="${CMD} +eval.loss_functions.noise_pred_sample_sizes=[${noise_pred_sample_sizes_str}]"
    CMD="${CMD} +eval.loss_functions.action_rec_sample_sizes=[${action_rec_sample_sizes_str}]"
}


# Dataset curation protocol: N/A.
declare -A demo_filter_episodes=(
    ["none"]=""
)
declare -A test_filter_episodes=(
    ["none"]=""
)
function add_experiment_cmd {
    if [[ ${crop_demo_episodes} == 1 ]]; then
        CMD="${CMD} +eval.max_demo_episode_length=${max_episode_length}"
    fi
    if [[ ${crop_test_episodes} == 1 ]]; then
        CMD="${CMD} +eval.max_test_episode_length=${max_episode_length}"
    fi
    
    if [[ ${calib_on_light} == 0 ]]; then
        echo "calib_on_light is a deprecated parameter than must be set to 1."
        exit 1
    else
        demo_experiment="${demo_date}_${demos}_${rollout_script}_${domain}_${agent}_${ckpt_name}_exec_horizon_${exec_horizon}_randomization_ca_${postfix_demo}"
        CMD="${CMD} +eval.demo_experiment=${demo_experiment} +eval.filter_demo_success=0 +eval.filter_demo_failure=1"
        
        # Curate demo/calibration episodes.
        if [[ -z "${demo_filter_episodes[${demo_experiment}]}" ]]; then
            CMD="${CMD} +eval.filter_demo_episodes=null"
        else
            local demo_episodes_str="${demo_filter_episodes[${demo_experiment}]}"
            demo_episodes_str="${demo_episodes_str// /,}"
            CMD="${CMD} +eval.filter_demo_episodes=[${demo_episodes_str}]"
        fi
    fi
    
    # Curate test episodes.
    if [[ -z "${test_filter_episodes[${prefix}]}" ]]; then
        CMD="${CMD} +eval.filter_test_episodes=null"
    else
        local test_episodes_str="${test_filter_episodes[${prefix}]}"
        test_episodes_str="${test_episodes_str// /,}"
        CMD="${CMD} +eval.filter_test_episodes=[${test_episodes_str}]"
    fi
}


function evaluate_ood_detectors {
    for agent in "${agents[@]}"; do
        for exec_horizon in "${exec_horizons[@]}"; do
            ckpt_path="${ckpt_paths[${domain}_${demos}_${agent}]}"
            ckpt_name="${ckpt_names[${domain}_${demos}_${agent}]}"
            max_episode_length="${max_episode_lengths[${domain}_${demos}_${agent}]}"
            max_num_episodes="${num_eval_episodes[${domain}_${demos}_${agent}]}"
            get_base_cmd

            if [[ ${evaluate_temporal_consistency} == 1 ]]; then
                CMD="${CMD} +eval.evaluate_temporal_consistency=1"
                add_temporal_consistency_cmd
            else
                CMD="${CMD} +eval.evaluate_temporal_consistency=0"
            fi

            if [[ ${evaluate_diffusion_ensemble} == 1 ]]; then
                CMD="${CMD} +eval.evaluate_diffusion_ensemble=1"
                add_diffusion_ensemble_cmd
            else
                CMD="${CMD} +eval.evaluate_diffusion_ensemble=0"
            fi

            if [[ ${evaluate_embedding_similarity} == 1 ]]; then
                CMD="${CMD} +eval.evaluate_embedding_similarity=1"
                add_embedding_similarity_cmd
            else
                CMD="${CMD} +eval.evaluate_embedding_similarity=0"
            fi

            if [[ ${evaluate_loss_functions} == 1 ]]; then
                CMD="${CMD} +eval.evaluate_loss_functions=1"
                add_loss_function_cmd
            else
                CMD="${CMD} +eval.evaluate_loss_functions=0"
            fi
        
            local quantiles_str="${quantiles[*]}"
            quantiles_str="${quantiles_str// /,}"
            CMD="${CMD} +eval.quantiles=[${quantiles_str}]"

            postfix_demo="rigid_pos_${randomize_rigid_pos}_rigid_scale_${randomize_rigid_scale}_soft_dynamics_${randomize_soft_dynamics}_robot_eef_pos_${randomize_robot_eef_pos}_dynamics_noise_${randomize_dynamics_noise}"

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

            postfix="rigid_pos_${randomize_rigid_pos}_rigid_scale_${randomize_rigid_scale}_soft_dynamics_${randomize_soft_dynamics}_robot_eef_pos_${randomize_robot_eef_pos}_dynamics_noise_${randomize_dynamics_noise}"
            prefix="${rollout_date}_${demos}_${rollout_script}_${domain}_${agent}_${ckpt_name}_exec_horizon_${exec_horizon}_randomization_${randomization}_${postfix}"
            add_experiment_cmd
            CMD="${CMD} +eval.calib_on_light=${calib_on_light} +eval.output_path=${date}_results_calib_on_light_${calib_on_light} prefix=${prefix}"

            run_cmd
        done
    done
}


function run_pusht {
    domain="pusht"
    crop_demo_episodes=1
    crop_test_episodes=1
    for randomization in "${randomizations[@]}"; do   
        randomize_rigid_pos=0
        randomize_rigid_scale=1
        randomize_robot_eef_pos=0
        randomize_soft_dynamics=0
        randomize_dynamics_noise=0
        evaluate_ood_detectors
    done
}


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
    ## Simulation demos: diffusion policy.
    ["pusht_sim_dp"]=0.90
    ## Simulation demos: diffusion policy.
    ["pusht_sim_dp_aug"]=0.90
    ## Simulation demos: SIM(3) diffusion policy.
    ["pusht_sim_sim3_dp"]=0.90
)

# Episodes.
declare -A num_eval_episodes=(
    ## Diffusion policy.
    ["pusht_sim_dp"]=50
    ## Diffusion policy with data augmentation.
    ["pusht_sim_dp_aug"]=50
    ## SIM(3) diffusion policy.
    ["pusht_sim_sim3_dp"]=50
)
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
quantiles=(
    # 0.50
    # 0.75
    # 0.85
    # 0.90
    0.95
)

# Synthetic domain parameters.
demos="sim"
sim_freq=1
scale_mode="real_src"
randomize_scale=1
randomize_rotation=0
batch_size=256
action_scale=1


# Experiment setup.
date="<enter_date>"
script="evaluate_ood_detectors"

# Important note: Within the context of evaluation, "demo" refers to the calibration dataset (i.e., the evaluation directory)
# containing the successful, in-distribution rollouts on which the failure detectors' detection thresholds will be calibrated. 
# Hence, "demo_date" corresponds to the date the calibration dataset was constructed via the compute_rollout_actions_pusht.<sh/py> 
# scripts. We provide those dates below for the official PushT task. Furthermore, "calib_on_light" is a deprecated parameter that 
# should always be set to 1, implying that the failure detectors should be calibrated on a dataset of policy rollouts instead of 
# the policy's training dataset of demonstrations (an unsuccessful, early-stage experimental setting).
calib_on_light=1                            # deprecated parameter.
rollout_script="compute_rollout_actions"    # dataset creating script.


#### Official experiment domains/datasets (as in https://arxiv.org/pdf/2410.04640). ####

# Note (All methods): All methods can only be evaluated on datasets that have been specifically generated to support their operation 
# (i.e., pre-computing and storing the necessary tensors). For the datasets below, we have correctly set the applicable methods. 
# However, if you wish to create your own experimental datasets using compute_rollout_actions_pusht.sh, please see the following table:
# -------------------------------------------------------------------------------------------------------------------
# | evaluate_ood_detectors_pusht.sh  |  compute_rollout_actions_pusht.sh                                            |
# -------------------------------------------------------------------------------------------------------------------
# | evaluate_temporal_consistency=1  |  N/A.                                                                        |
# | evaluate_diffusion_ensemble=1    |  N/A.                                                                        |
# | evaluate_embedding_similarity=1  |  save_encoder_embeddings=1, save_resnet_embeddings=1, save_clip_embeddings=1 |
# | evaluate_loss_functions=1        |  save_noise_preds=1, save_rec_actions=1                                      |
# -------------------------------------------------------------------------------------------------------------------

## Experiment: PushT. ##
demo_date="0525"

# PushT main result (Figure 5): All methods sans VLM.
evaluate_temporal_consistency=1
evaluate_diffusion_ensemble=1
evaluate_embedding_similarity=1
evaluate_loss_functions=1
exec_horizons=(
    8
)
randomizations=(
    "na"
    "hh"
)
for rollout_date in "0525" "0526" "0527"; do
    run_pusht
done

# PushT ablation result (Figure 8, Figure 9): STAC.
evaluate_temporal_consistency=1
evaluate_diffusion_ensemble=0
evaluate_embedding_similarity=0
evaluate_loss_functions=0
exec_horizons=(
    2 
    4 
)
randomizations=(
    "na"
    "hh"
)
rollout_date="0525"
run_pusht