python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    data.path=/root/autodl-tmp/gsm8k/test1.parquet \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path=/root/autodl-tmp/gsm8k/output.parquet \
    model.path=/root/autodl-tmp/Llama-3.2-1B \
    +model.trust_remote_code=True \
    rollout.temperature=1.0 \
    rollout.top_k=50 \
    rollout.top_p=0.7 \
    rollout.prompt_length=2048 \
    rollout.response_length=1024 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.6 \
    # rollout.micro_batch_size=64
