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
    CMD="python -m sentinel.bc.${script} --config-name ${domain}_mobile_${agent} mode=eval seed=${seed}"
    CMD="${CMD} training.ckpt=${ckpt_path}/${ckpt_name}.pth training.batch_size=${batch_size}"
    CMD="${CMD} env.args.ac_scale=${action_scale} model.ac_horizon=${exec_horizon}"
    CMD="${CMD} env.args.scale_mode=${scale_mode} env.args.deform_randomize_scale=${randomize_scale}"
    CMD="${CMD} env.args.max_episode_length=${max_episode_length} +eval.max_num_episodes=${max_num_episodes}"
    CMD="${CMD} env.args.freq=${sim_freq} data.dataset_class=episode_dataset"
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
        32
        # 64
        # 128
        # 256
        # 512
    )
    local error_fns=(
        # Min.
        "mse_all"
        # "mse_pos"
        # "ate_pos"
        # MMD.
        "mmd_rbf_all"
        # "mmd_rbf_all_median"
        # "mmd_rbf_all_eig"
        # "mmd_rbf_all_0.1"
        # "mmd_rbf_all_0.5"
        # "mmd_rbf_all_1.0"
        # "mmd_rbf_all_5.0"
        # "mmd_rbf_all_10.0" 
        # "mmd_rbf_all_100.0" 
        # KDE For. KL.
        "kde_kl_all_for"
        # "kde_kl_all_for_eig"
        # "kde_kl_all_for_0.1"
        # "kde_kl_all_for_0.5"
        # "kde_kl_all_for_5.0"
        # "kde_kl_all_for_10.0"
        # "kde_kl_all_for_100.0"
        # KDE Rev. KL.
        "kde_kl_all_rev"
        # "kde_kl_all_rev_eig"
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


## Method: Vision-language model (VLM) runtime monitor.
# Note: Configuration below uses official hyperparameters.
function add_vlm_cmd {
    # Prompt Templates:
    # "image_qa": Reason over single image with task-specific context (questions, time).
    # "video_qa": Reason over video with task-specific context (questions, time).
    # "video_qa_ref_video": video_qa + in-context successful video.
    # "video_qa_ref_goal": video_qa + in-context successful goal states.
    local models=(
        "gpt-4o"
        "claude-3-5-sonnet-20240620"
        "gemini-1-5-pro"
    )
    declare -A templates=(
        ## Cover Object.
        ["cover_gpt-4o"]="image_qa,video_qa"
        ["cover_claude-3-5-sonnet-20240620"]="image_qa,video_qa,video_qa_ref_video,video_qa_ref_goal"
        ["cover_gemini-1-5-pro"]="image_qa,video_qa"
        ## Close Box.
        ["close_gpt-4o"]="image_qa,video_qa"
        ["close_claude-3-5-sonnet-20240620"]="image_qa,video_qa"
        ["close_gemini-1-5-pro"]="image_qa,video_qa"
    )
    declare -A num_timesteps=(
        ["cover"]=2
        ["close"]=2
    )
    declare -A subsample_freq=(
        ["cover"]=2
        ["close"]=1
    )
    declare -A max_video_length=(
        ["cover"]=64
        ["close"]=120
    )
    declare -A ref_experiment=(
        ["cover"]="0826_sim_compute_rollout_actions_cover_dp_ckpt00999_exec_horizon_4_randomization_na_rigid_pos_1_rigid_scale_0_soft_dynamics_0_robot_eef_pos_1_dynamics_noise_0"
        ["close"]="0826_sim_compute_rollout_actions_close_dp_ckpt00999_exec_horizon_4_randomization_na_rigid_pos_0_rigid_scale_1_soft_dynamics_0_robot_eef_pos_1_dynamics_noise_0"
    )
    
    local models_str="${models[*]}"
    models_str="${models_str// /,}"

    CMD="${CMD} +eval.vlm.domain=${domain}"
    CMD="${CMD} +eval.vlm.models=[${models_str}]"
    CMD="${CMD} +eval.vlm.max_video_length=${max_video_length[${domain}]}"
    CMD="${CMD} +eval.vlm.num_timesteps=${num_timesteps[${domain}]}"
    CMD="${CMD} +eval.vlm.subsample_freq=${subsample_freq[${domain}]}"
    CMD="${CMD} +eval.vlm.prompt_dir=${PWD}/sentinel/bc/configs/prompts/official"
    CMD="${CMD} +eval.vlm.ref_experiment=${ref_experiment[${domain}]}"

    # Add prompt templates.
    for model in "${models[@]}"; do
        if [[ "${model}" == "gpt-4o" && -z "${OPENAI_API_KEY}" ]]; then
            echo "Error: OPENAI_API_KEY is not set."
            exit 1
        elif [[ "${model}" == "claude-3-5-sonnet-20240620" && -z "${ANTHROPIC_API_KEY}" ]]; then
            echo "Error: ANTHROPIC_API_KEY is not set."
            exit 1
        elif [[ "${model}" == "gemini-1-5-pro" && -z "${GOOGLE_API_KEY}" ]]; then
            echo "Error: GOOGLE_API_KEY is not set."
            exit 1
        fi

        key="${domain}_${model}"
        if [[ -z "${templates[${key}]}" ]]; then
            echo "Prompt template key ${key} is undefined."
            exit 1
        fi
        CMD="${CMD} +eval.vlm.templates.${model}=[${templates[${key}]}]"
    done
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
        32 
        # 64 
        # 128
        # 256
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


# Dataset curation protocol:
# Cover Object: Filter episodes in which the deformable cloth explodes due to simulation instability.
# Close Box: Filter episodes of a) task progression failures in erratic failure data splits; b) erratic 
# failures in task progression failure data splits; c) episodes in both the calibration splits and the 
# test splits where the policy succeeds but jitters excessively (which we uniformly consider as failing
# behavior, and hence, we do not calibrate the failure detector on such episodes. We filter such episodes
# from the test splits to avoid incorrectly labeled false positives, where the policy succeeds with excessive 
# jitter, and the failure detector raises a "false" alarm, being not calibrated on such episodes. Note that
# such episodes could analogously be added back to both the calibration and test splits).
declare -A demo_filter_episodes=(
    # Close calibration dataset.
    ["0525_sim_compute_rollout_actions_close_dp_ckpt00999_exec_horizon_4_randomization_ca_rigid_pos_0_rigid_scale_1_soft_dynamics_0_robot_eef_pos_1_dynamics_noise_0"]="6 13 14 23 35"
    # Cover calibration dataset.
    ["0527_sim_compute_rollout_actions_cover_dp_ckpt00999_exec_horizon_4_randomization_ca_rigid_pos_1_rigid_scale_0_soft_dynamics_0_robot_eef_pos_1_dynamics_noise_0"]="22 24 30 31 32 39 41"
)
declare -A test_filter_episodes=(
    #### Archived datasets. ####

    # Close hyperparameter sweep datasets.
    ["0525_sim_compute_rollout_actions_close_dp_ckpt00999_exec_horizon_4_randomization_na_rigid_pos_0_rigid_scale_1_soft_dynamics_0_robot_eef_pos_1_dynamics_noise_0"]="7 30 46"
    ["0525_sim_compute_rollout_actions_close_dp_ckpt00999_exec_horizon_4_randomization_ll_rigid_pos_0_rigid_scale_1_soft_dynamics_0_robot_eef_pos_1_dynamics_noise_0"]="6 26"
    ["0525_sim_compute_rollout_actions_close_dp_ckpt00999_exec_horizon_4_randomization_hh_rigid_pos_0_rigid_scale_1_soft_dynamics_0_robot_eef_pos_1_dynamics_noise_0"]="5 8 10 22 25 46 49"
    
    # Close, Cover visualization datasets.
    ["0525_sim_compute_rollout_actions_close_dp_ckpt00999_exec_horizon_4_randomization_ss_rigid_pos_0_rigid_scale_1_soft_dynamics_0_robot_eef_pos_1_dynamics_noise_0"]="13 30"
    ["0527_sim_compute_rollout_actions_cover_dp_ckpt00999_exec_horizon_4_randomization_ss_rigid_pos_1_rigid_scale_0_soft_dynamics_0_robot_eef_pos_1_dynamics_noise_0"]="3 7 8 11 12 14 15 21 22 23 25 28 30 34 35 44 46"

    #### Official result datasets. ####

    # Close erratic failures (Table 1, Table 5): All methods sans VLM.
    ["0527_sim_compute_rollout_actions_close_dp_ckpt00999_exec_horizon_4_randomization_na_rigid_pos_0_rigid_scale_1_soft_dynamics_0_robot_eef_pos_1_dynamics_noise_0"]="20 21 29 43"
    ["0527_sim_compute_rollout_actions_close_dp_ckpt00999_exec_horizon_4_randomization_hh_rigid_pos_0_rigid_scale_1_soft_dynamics_0_robot_eef_pos_1_dynamics_noise_0"]="1 12 16 23 30 31 32 37 44"
    ["0528_sim_compute_rollout_actions_close_dp_ckpt00999_exec_horizon_4_randomization_na_rigid_pos_0_rigid_scale_1_soft_dynamics_0_robot_eef_pos_1_dynamics_noise_0"]="2 17 18 22 24 25 26 35 43 44"
    ["0528_sim_compute_rollout_actions_close_dp_ckpt00999_exec_horizon_4_randomization_hh_rigid_pos_0_rigid_scale_1_soft_dynamics_0_robot_eef_pos_1_dynamics_noise_0"]="14 26 36 43"
    ["0529_sim_compute_rollout_actions_close_dp_ckpt00999_exec_horizon_4_randomization_na_rigid_pos_0_rigid_scale_1_soft_dynamics_0_robot_eef_pos_1_dynamics_noise_0"]="4 8 20 24"
    ["0529_sim_compute_rollout_actions_close_dp_ckpt00999_exec_horizon_4_randomization_hh_rigid_pos_0_rigid_scale_1_soft_dynamics_0_robot_eef_pos_1_dynamics_noise_0"]="9 13 41 48"
    
    # Close erratic failures (Table 1, Table 5): STAC and VLM (Sentinel).
    ["0826_sim_compute_rollout_actions_close_dp_ckpt00999_exec_horizon_4_randomization_na_rigid_pos_0_rigid_scale_1_soft_dynamics_0_robot_eef_pos_1_dynamics_noise_0"]="4 7 10 11 13 14 30 40 43"
    ["0826_sim_compute_rollout_actions_close_dp_ckpt00999_exec_horizon_4_randomization_hh_rigid_pos_0_rigid_scale_1_soft_dynamics_0_robot_eef_pos_1_dynamics_noise_0"]="0 2 13 15 18 24 26 31 32 36 37 38 45 49"

    # Close task progression failures (Figure 6, Table 7): STAC and VLM (Sentinel).    
    ["0826_sim_compute_rollout_actions_close_dp_ckpt00999_exec_horizon_4_randomization_ss_rigid_pos_0_rigid_scale_1_soft_dynamics_0_robot_eef_pos_1_dynamics_noise_0"]="9 17 30 32"

    # Cover task progression failures (Figure 6, Table 6): STAC and VLM (Sentinel).    
    ["0826_sim_compute_rollout_actions_cover_dp_ckpt00999_exec_horizon_4_randomization_na_rigid_pos_1_rigid_scale_0_soft_dynamics_0_robot_eef_pos_1_dynamics_noise_0"]="9 19 38 40 44 48"
    ["0904_sim_compute_rollout_actions_cover_dp_ckpt00999_exec_horizon_4_randomization_ss_rigid_pos_1_rigid_scale_0_soft_dynamics_0_robot_eef_pos_1_dynamics_noise_0"]="1 2 3 9 13 17 20 21 22 24 28 31 34 36 37 46"
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

            if [[ ${evaluate_vlm} == 1 ]]; then
                CMD="${CMD} +eval.evaluate_vlm=1"
                add_vlm_cmd
            else
                CMD="${CMD} +eval.evaluate_vlm=0"
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


function run_cover {
    domain="cover"
    crop_demo_episodes=1
    crop_test_episodes=1
    for randomization in "${randomizations[@]}"; do
        randomize_rigid_pos=1
        randomize_rigid_scale=0
        randomize_robot_eef_pos=1
        randomize_soft_dynamics=0
        randomize_dynamics_noise=0
        evaluate_ood_detectors
    done
}


function run_close {
    domain="close"
    crop_demo_episodes=1
    crop_test_episodes=1
    for randomization in "${randomizations[@]}"; do
        randomize_rigid_pos=0
        randomize_rigid_scale=1
        randomize_robot_eef_pos=1
        randomize_soft_dynamics=0
        randomize_dynamics_noise=0
        evaluate_ood_detectors
    done
}


function run_fold {
    domain="fold"
    crop_demo_episodes=1
    crop_test_episodes=1
    for randomization in "${randomizations[@]}"; do   
        randomize_rigid_pos=0
        randomize_rigid_scale=0
        randomize_robot_eef_pos=1
        randomize_soft_dynamics=1
        randomize_dynamics_noise=0
        evaluate_ood_detectors
    done
}


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

# Episodes.
declare -A num_eval_episodes=(
    ## Diffusion policy.
    ["fold_sim_dp"]=30
    ["cover_sim_dp"]=50
    ["close_sim_dp"]=50
    ## SIM(3) diffusion policy.
    ["fold_sim_sim3_dp"]=30
    ["cover_sim_sim3_dp"]=30
    ["close_sim_sim3_dp"]=30
)
declare -A max_episode_lengths=(
    ## Diffusion policy.
    ["fold_sim_dp"]=50
    ["cover_sim_dp"]=45
    ["close_sim_dp"]=100
    ## SIM(3) diffusion policy.
    ["fold_sim_sim3_dp"]=50
    ["cover_sim_sim3_dp"]=50
    ["close_sim_sim3_dp"]=100
)

# Experiment parameters.
seed=0
exec_horizons=(
    # 2
    4 
    # 8
)
quantiles=(
    # 0.50
    # 0.75
    # 0.85
    # 0.90
    0.95
    # 0.98
    # 0.99
)

# Mobile domain parameters.
demos="sim"
sim_freq=5
scale_mode="real_src"
randomize_scale=1
batch_size=256
action_scale=1


# Experiment setup.
date="<enter_date>"
script="evaluate_ood_detectors"

# Important note: Within the context of evaluation, "demo" refers to the calibration dataset (i.e., the evaluation directory)
# containing the successful, in-distribution rollouts on which the failure detectors' detection thresholds will be calibrated. 
# Hence, "demo_date" corresponds to the date the calibration dataset was constructed via the compute_rollout_actions_mobile.<sh/py> 
# scripts. We provide those dates below for the official Close Box and Cover Object tasks. Furthermore, "calib_on_light" is a 
# deprecated parameter that should always be set to 1, implying that the failure detectors should be calibrated on a dataset of 
# policy rollouts instead of the policy's training dataset of demonstrations (an unsuccessful, early-stage experimental setting).
calib_on_light=1                            # deprecated parameter.
rollout_script="compute_rollout_actions"    # dataset creating script.


#### Official experiment domains/datasets (as in https://arxiv.org/pdf/2410.04640). ####

# Note (VLMs): Evaluating VLM monitors (evaluate_vlm=1) requires certain environment variables to be set. 
# Please add the following to your ~/.bashrc:
# export OPENAI_API_KEY=...
# export ANTHROPIC_API_KEY=...
# export GOOGLE_API_KEY=...

# Note (All other methods): All other methods can only be evaluated on datasets that have been specifically generated to support their
# operation (i.e., pre-computing and storing the necessary tensors). For the datasets below, we have correctly set the applicable methods.
# However, if you wish to create your own experimental datasets using compute_rollout_actions_mobile.sh, please see the following table:
# --------------------------------------------------------------------------------------------------------------------
# | evaluate_ood_detectors_mobile.sh  |  compute_rollout_actions_mobile.sh                                           |
# --------------------------------------------------------------------------------------------------------------------
# | evaluate_temporal_consistency=1   |  N/A.                                                                        |
# | evaluate_diffusion_ensemble=1     |  N/A.                                                                        |
# | evaluate_embedding_similarity=1   |  save_encoder_embeddings=1, save_resnet_embeddings=1, save_clip_embeddings=1 |
# | evaluate_loss_functions=1         |  save_noise_preds=1, save_rec_actions=1                                      |
# --------------------------------------------------------------------------------------------------------------------


## Experiment: Close. ##
demo_date="0525"

# Erratic failures (Table 1, Table 5): All methods sans VLM.
evaluate_temporal_consistency=1
evaluate_diffusion_ensemble=1
evaluate_embedding_similarity=1
evaluate_loss_functions=1
evaluate_vlm=0
randomizations=(
    "na"
    "hh"
)
for rollout_date in "0527" "0528" "0529"; do
    run_close
done

# Erratic failures (Table 1, Table 5): STAC and VLM (Sentinel).
evaluate_temporal_consistency=1
evaluate_diffusion_ensemble=1
evaluate_embedding_similarity=0
evaluate_loss_functions=0
evaluate_vlm=1
randomizations=(
    "na"
    "hh"
)
rollout_date="0826"
run_close

# Task progression failures (Figure 6, Table 7): STAC and VLM (Sentinel).
evaluate_temporal_consistency=1
evaluate_diffusion_ensemble=1
evaluate_embedding_similarity=0
evaluate_loss_functions=0
evaluate_vlm=1
randomizations=(
    "ss"
)
rollout_date="0826"
run_close

## Experiment: Cover. ##
demo_date="0527" 

# Task progression failures (Figure 6, Table 6): STAC and VLM (Sentinel).
evaluate_temporal_consistency=1
evaluate_diffusion_ensemble=1
evaluate_embedding_similarity=0
evaluate_loss_functions=0
evaluate_vlm=1
randomizations=(
    "na"
)
rollout_date="0826"
run_cover
randomizations=(
    "ss"
)
rollout_date="0904"
run_cover


#### Unofficial domain, included for completion and prototyping. ####

## Experiment: Fold. ##
# demo_date="<enter>"
# evaluate_temporal_consistency=0
# evaluate_diffusion_ensemble=0
# evaluate_embedding_similarity=0
# evaluate_loss_functions=0
# evaluate_vlm=0
# randomizations=(
#     "na"
#     "ll"
#     "hh"
# )
# rollout_date="<enter>"
# run_fold