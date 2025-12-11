set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m examples.solver_judge_skyrl.train_solver_judge \
    data.train_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    generator.model.path=Qwen/Qwen3-4B-Instruct-2507 \
    generator.sampling_params.temperature=1.0 \
    generator.sampling_params.top_p=1.0 \
    generator.sampling_params.max_tokens=1024 \
    generator.n_parallel_tasks=128 \
    generator.retry_limit=3 \
    rllm.workflow.use_workflow=True \
    rllm.workflow.n_parallel_tasks=128 \
    rllm.workflow.retry_limit=3 \
    rllm.stepwise_advantage.enable=True \
    rllm.stepwise_advantage.mode=per_step \
    rllm.compact_filtering.enable=False \
    rllm.compact_filtering.mask_max_prompt_length_exceeded=True \
    rllm.compact_filtering.mask_max_response_length_exceeded=True \
    rllm.compact_filtering.mask_max_turns_exceeded=False \
    rllm.compact_filtering.mask_timeout=True \
    rllm.rejection_sample.enable=False \
    rllm.rejection_sample.multiplier=1.0 \
    algorithm.adv_estimator=grpo \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='solver-judge-workflow-skyrl' \
    trainer.experiment_name='countdown-solver-judge-skyrl' \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=1000 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=100 \
    +ray_init._temp_dir=/tmp

pkill -9 -f 'ray::WorkerDict'

