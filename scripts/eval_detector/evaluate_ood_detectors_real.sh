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
    CMD="python -m sentinel.bc.${script} --config-name ${robot_setup}_${agent} robot_info=${domain} mode=eval seed=${seed}"
    CMD="${CMD} robot_info.use_dummy_zed=true robot_info.freq=${robot_freq} +env.args.freq=${robot_freq}"
    CMD="${CMD} training.ckpt=${ckpt_path}/${ckpt_name}.pth training.batch_size=${batch_size} model.ac_horizon=${exec_horizon}"
    CMD="${CMD} env.args.max_episode_length=${max_episode_length} +eval.max_num_episodes=${max_num_episodes} env.args.ac_scale=${action_scale}"
    CMD="${CMD} data.dataset_class=episode_dataset log_dir=${log_dir} use_wandb=false"
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
    CMD="${CMD} +eval.consistency.skip_steps=${skip_steps}"
}


## Method: Vision-language model (VLM) runtime monitor.
# Note: Configuration below uses official hyperparameters.
function add_vlm_cmd {
    # Prompt Templates:
    # "image": Reason over single image.
    # "video": Reason over video (sequence of images).
    # "video_task": Reason over video with task-specific context (questions, time).
    # "video_context": Reason over video with in-context video as a reference for success.
    local models=(
        "gpt-4o"
        "claude-3-5-sonnet-20240620"
        "gemini-1-5-pro"
    )
    declare -A templates=(
        ## Push Chair.
        ["push_chair_gpt-4o"]="video_qa"
        ["push_chair_claude-3-5-sonnet-20240620"]="video_qa"
        ["push_chair_gemini-1-5-pro"]="video_qa"
    )
    declare -A num_timesteps=(
        ["push_chair"]=2
    )
    declare -A subsample_freq=(
        ["push_chair"]=1
    )
    declare -A max_video_length=(
        ["push_chair"]=48
    )
    declare -A ref_experiment=(
        ["push_chair"]="0914_push_chair_sim3_dp_ddim_calib"
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
    
    demo_experiment="${demo_date}_${domain}_${agent}_calib"
    CMD="${CMD} +eval.demo_experiment=${demo_experiment} +eval.filter_demo_success=0 +eval.filter_demo_failure=1"
    
    # Curate demo/calibration episodes.
    if [[ -z "${demo_filter_episodes[${demo_experiment}]}" ]]; then
        CMD="${CMD} +eval.filter_demo_episodes=null"
    else
        local demo_episodes_str="${demo_filter_episodes[${demo_experiment}]}"
        demo_episodes_str="${demo_episodes_str// /,}"
        CMD="${CMD} +eval.filter_demo_episodes=[${demo_episodes_str}]"
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
            ckpt_path="${ckpt_paths[${domain}_${agent}]}"
            ckpt_name="${ckpt_names[${domain}_${agent}]}"
            max_episode_length="${max_episode_lengths[${domain}_${agent}]}"
            max_num_episodes="${num_eval_episodes[${domain}_${agent}]}"
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

            prefix="${rollout_date}_${domain}_${agent}_test"
            add_experiment_cmd
            CMD="${CMD} +eval.calib_on_light=${calib_on_light} +eval.output_path=${date}_results_calib_on_light_${calib_on_light} prefix=${prefix}"

            run_cmd
        done
    done
}

function run_push_chair {
    domain="push_chair"
    robot_setup="real_single_arm"
    crop_demo_episodes=1
    crop_test_episodes=1
    evaluate_ood_detectors
}


# Agents.
agents=(
    ## Diffusion policy.
    # "dp_ddim"
    ## SIM(3) diffusion policy.
    "sim3_dp_ddim"
)

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
declare -A num_eval_episodes=(
    ## Diffusion policy.
    ["push_chair_dp_ddim"]=20
    ## SIM(3) diffusion policy.
    ["push_chair_sim3_dp_ddim"]=20
)
declare -A max_episode_lengths=(
    ## Diffusion policy.
    ["push_chair_dp_ddim"]=40
    ## SIM(3) diffusion policy.
    ["push_chair_sim3_dp_ddim"]=40
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

# Real domain parameters.
batch_size=256
robot_freq=3
action_scale=3
skip_steps=1


# Experiment setup.
date="<enter_date>"
script="evaluate_ood_detectors"
log_dir="logs/bc/real_eval"

# Important note: Within the context of evaluation, "demo" refers to the calibration dataset (i.e., the evaluation directory)
# containing the successful, in-distribution rollouts on which the failure detectors' detection thresholds will be calibrated. 
# Hence, "demo_date" corresponds to the date the calibration dataset was collected on the real-robot platform (unlike the sim 
# domains, we cannot synthetically create rollouts for real-world experiments, and hence compute_rollout_actions_real.<sh/py>
# serves a different purpose: to convert real-world log data into tensors necessary to run our failure detectors). We provide 
# the demo dates below for the official Push Chair task. Furthermore, "calib_on_light" is a deprecated parameter that should 
# always be set to 1, implying that the failure detectors should be calibrated on a dataset of policy rollouts instead of the 
# policy's training dataset of demonstrations (an unsuccessful, early-stage experimental setting).
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
# However, if you wish to create your own experimental datasets using compute_rollout_actions_real.sh, please see the following table:
# --------------------------------------------------------------------------------------------------------------------
# | evaluate_ood_detectors_real.sh   |  compute_rollout_actions_real.sh                                              |
# --------------------------------------------------------------------------------------------------------------------
# | evaluate_temporal_consistency=1  |  N/A.                                                                         |
# | evaluate_diffusion_ensemble=1    |  N/A.                                                                         |
# | evaluate_embedding_similarity=1  |  save_encoder_embeddings=1, save_resnet_embeddings=1, save_clip_embeddings=1  |
# | evaluate_loss_functions=1        |  save_noise_preds=1, save_rec_actions=1 (not supported for SIM(3) agents)     |
# --------------------------------------------------------------------------------------------------------------------

## Experiment: Push chair. ##
demo_date="0914"

# Push chair real-world result (Table 2): STAC and VLM (Sentinel).
evaluate_temporal_consistency=1
evaluate_diffusion_ensemble=1
evaluate_embedding_similarity=0
evaluate_loss_functions=0
evaluate_vlm=1
rollout_date="0914"
run_push_chair