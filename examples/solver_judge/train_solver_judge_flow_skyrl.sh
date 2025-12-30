set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export NUM_GPUS=1
export DATA_DIR="/home/ray/rllm/~/data/rlhf/gsm8k"
export LOGGER="wandb"
export INFERENCE_BACKEND="vllm"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#uv run --isolated --extra vllm 
python3 -m examples.solver_judge.train_solver_judge_flow \
    data.train_data="['$DATA_DIR/train.parquet']" \
    data.val_data="['$DATA_DIR/validation.parquet']" \
    trainer.algorithm.advantage_estimator="grpo" \
    trainer.policy.model.path="Qwen/Qwen2.5-0.5B" \
    trainer.placement.colocate_all=true \
    trainer.strategy=fsdp2 \
    trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
    trainer.placement.critic_num_gpus_per_node=$NUM_GPUS \
    trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
    generator.num_inference_engines=$NUM_GPUS \
    generator.inference_engine_tensor_parallel_size=1 \
    trainer.epochs=20 \
    trainer.eval_batch_size=1024 \
    trainer.eval_before_train=true \
    trainer.eval_interval=5 \
    trainer.update_epochs_per_batch=1 \
    trainer.train_batch_size=1024 \
    trainer.policy_mini_batch_size=256 \
    trainer.micro_forward_batch_size_per_gpu=1 \
    trainer.micro_train_batch_size_per_gpu=1 \
    trainer.ckpt_interval=1 \
    trainer.max_prompt_length=512 \
    generator.sampling_params.max_generate_length=512 \
    trainer.policy.optimizer_config.lr=1.0e-6 \
    trainer.algorithm.use_kl_loss=true \
    generator.backend=$INFERENCE_BACKEND \
    generator.run_engines_locally=true \
    generator.weight_sync_backend=nccl \
    generator.async_engine=true \
    generator.batched=true \
    environment.env_class=gsm8k \
    generator.n_samples_per_prompt=5 \
    generator.gpu_memory_utilization=0.92 \
    trainer.logger=$LOGGER \
    trainer.project_name="gsm8k" \
    trainer.run_name="gsm8k_test" \
    trainer.resume_mode=null \
    trainer.ckpt_path="$HOME/ckpts/gsm8k_deepspeed_retrain" \
    rllm.compact_filtering.enable=False \
    rllm.compact_filtering.mask_max_prompt_length_exceeded=True \
    rllm.compact_filtering.mask_max_response_length_exceeded=True \
    rllm.compact_filtering.mask_max_turns_exceeded=False \
    rllm.compact_filtering.mask_timeout=True \
    rllm.rejection_sample.enable=False \
    rllm.rejection_sample.multiplier=1.0 \
    rllm.stepwise_advantage.enable=True \
    rllm.stepwise_advantage.mode=per_step \
    rllm.workflow.use_workflow=True \
    ray_init._temp_dir=/home/ray/tmp

pkill -9 -f 'ray::WorkerDict' 