# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/worker/worker.py
"""A GPU worker class."""
import gc
import os
import time
from typing import Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torch.distributed
import torch.nn as nn

import vllm.envs as envs
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment,
                              set_custom_all_reduce)
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import (ExecuteModelRequest, IntermediateTensors,
                           SequenceGroupMetadata, SequenceGroupMetadataDelta)
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.embedding_model_runner import EmbeddingModelRunner
from vllm.worker.enc_dec_model_runner import EncoderDecoderModelRunner
from vllm.worker.model_runner import GPUModelRunnerBase, ModelRunner
from vllm.worker.worker_base import (LocalOrDistributedWorkerBase, WorkerBase,
                                     WorkerInput)
# TODO(sgm): check why vllm has similar file in vllm.model_executor.parallel_utils.parallel_state
from vllm.distributed import (init_distributed_environment, set_custom_all_reduce, get_tensor_model_parallel_group)
from vllm.worker.worker import Worker, _check_if_gpu_supports_dtype
from vllm.worker.model_runner_base import ModelRunnerBase, ModelRunnerInputBase

from .model_runner import ModelRunner
from .megatron_weight_loaders import load_megatron_weights
from .hf_weight_loader import load_hf_weights
from .dtensor_weight_loaders import load_dtensor_weights
from .parallel_state import (ensure_model_parallel_initialized)
from .config import ModelConfig, LoadConfig, LoadFormat

logger = init_logger(__name__)







class Worker(Worker):
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        model: Union[nn.Module, Dict], # model itself or its parameter dict
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
        model_runner_cls: Optional[Type[GPUModelRunnerBase]] = None,
    ) -> None:
        WorkerBase.__init__(self, vllm_config)
        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker
        # if is_driver_worker:
        #     assert rank % self.parallel_config.tensor_parallel_size == 0, \
        #            "Driver worker should be rank 0 of tensor parallel group."
        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

        # Return hidden states from target model if the draft model is an
        # mlp_speculator
        speculative_config = self.speculative_config
        model_config = self.model_config
        speculative_args = {} if speculative_config is None \
            or (speculative_config.draft_model_config.model ==
                model_config.model) \
            or (speculative_config.draft_model_config.hf_config.model_type
                not in ["medusa", "mlp_speculator", "eagle"]) \
                    else {"return_hidden_states": True}

        ModelRunnerClass: Type[GPUModelRunnerBase] = ModelRunner
        if model_runner_cls is not None:
            ModelRunnerClass = model_runner_cls
        elif model_config.task == "embedding":
            ModelRunnerClass = EmbeddingModelRunner
        elif self.model_config.is_encoder_decoder:
            ModelRunnerClass = EncoderDecoderModelRunner
        self.model_runner: GPUModelRunnerBase = ModelRunnerClass(
            model, # [VERL]: add for verl
            vllm_config=self.vllm_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=is_driver_worker,
            **speculative_args,
        )
        # Uninitialized cache engine. Will be initialized by
        # initialize_cache.
        self.cache_engine: List[CacheEngine]
        # Initialize gpu_cache as embedding models don't initialize kv_caches
        self.gpu_cache: Optional[List[List[torch.Tensor]]] = None
        self._seq_group_metadata_cache: Dict[str, SequenceGroupMetadata] = {}

        # NOTE(sgm): [VERL] For offloading inference engine params
        self.cpu_model = None

        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        if envs.VLLM_TORCH_PROFILER_DIR:
            torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            logger.info("Profiling enabled. Traces will be saved to: %s",
                        torch_profiler_trace_dir)
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    torch_profiler_trace_dir, use_gzip=True))
        else:
            self.profiler = None

    def init_device(self) -> None:
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # NOTE(sgm): Modify for verl, Env vars will be set by TORCHRUN.
            self.rank = self.rank if self.rank is not None else int(os.getenv("RANK", "-1"))
            local_rank = int(os.getenv("LOCAL_RANK", "0"))
            self.device = torch.device(f"cuda:{local_rank}")
            if self.rank < 0:
                raise ValueError("Invalid or unspecified rank.")
            torch.cuda.set_device(self.device)

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            gc.collect()
            torch.cuda.empty_cache()
            self.init_gpu_memory = torch.cuda.mem_get_info()[0]
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        init_worker_distributed_environment(self.parallel_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank)
        # Set random seed.
        set_random_seed(self.model_config.seed)    

    @torch.inference_mode()
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Profiles the peak memory usage of the model to determine how many
        KV blocks may be allocated without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.

        .. tip::
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()
        # torch.cuda.reset_peak_memory_stats()

        free_memory_pre_profile, total_gpu_memory = torch.cuda.mem_get_info()
        start_time = time.time()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        self.model_runner.profile_run()
        torch.cuda.synchronize()

        self._assert_memory_footprint_increased_during_profiling()

        # Get the peak memory allocation recorded by torch
        peak_memory = torch.cuda.memory_stats()["allocated_bytes.all.peak"]

        # Check for any memory left around that may have been allocated on the
        # gpu outside of `torch`. NCCL operations, for example, can use a few
        # GB during a forward pass
        torch.cuda.empty_cache()
        torch_allocated_bytes = torch.cuda.memory_stats(
        )["allocated_bytes.all.current"]
        total_allocated_bytes = torch.cuda.mem_get_info(
        )[1] - torch.cuda.mem_get_info()[0]
        non_torch_allocations = total_allocated_bytes - torch_allocated_bytes
        if non_torch_allocations > 0:
            peak_memory += non_torch_allocations

        available_kv_cache_memory = (
            total_gpu_memory * self.cache_config.gpu_memory_utilization -
            peak_memory)

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        cache_block_size = self.get_cache_block_size_bytes()
        if cache_block_size == 0:
            num_gpu_blocks = 0
            num_cpu_blocks = 0
        else:
            num_gpu_blocks = int(available_kv_cache_memory // cache_block_size)
            num_cpu_blocks = int(self.cache_config.swap_space_bytes //
                                 cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)

        end_time = time.time()
        logger.info(
            "Memory profiling results: "
            "duration=%.2f seconds, "
            "total_gpu_memory=%.2fGiB, "
            "initial_memory_usage=%.2fGiB, "
            "peak_torch_memory=%.2fGiB, "
            "memory_usage_post_profile=%.2fGiB, "
            "non_torch_memory=%.2fGiB, "
            "kv_cache_size=%.2fGiB, "
            "gpu_memory_utilization=%.2f.", end_time - start_time,
            total_gpu_memory / (1024**3),
            (total_gpu_memory - free_memory_pre_profile) / (1024**3),
            (peak_memory - non_torch_allocations) / (1024**3),
            total_allocated_bytes / (1024**3),
            non_torch_allocations / (1024**3),
            available_kv_cache_memory / (1024**3),
            self.cache_config.gpu_memory_utilization)

        # Final cleanup
        if self.model_runner.lora_manager:
            self.model_runner.remove_all_loras()
        
        # NOTE(sgm): Add for [VERL], synchronize number of blocks with all the rank
        num_gpu_blocks = torch.tensor([num_gpu_blocks], device='cuda')
        num_cpu_blocks = torch.tensor([num_cpu_blocks], device='cuda')

        torch.distributed.all_reduce(num_gpu_blocks,
                                     op=torch.distributed.ReduceOp.MIN,
                                     group=get_tensor_model_parallel_group().device_group)
        torch.distributed.all_reduce(num_cpu_blocks,
                                     op=torch.distributed.ReduceOp.MIN,
                                     group=get_tensor_model_parallel_group().device_group)
        num_gpu_blocks = num_gpu_blocks.item()
        num_cpu_blocks = num_cpu_blocks.item()

        gc.collect()

        return num_gpu_blocks, num_cpu_blocks

    def _init_cache_engine(self):
        assert self.cache_config.num_gpu_blocks is not None
        self.cache_engine = [
            CacheEngine(self.cache_config, self.model_config,
                        self.parallel_config, self.device_config)
            for _ in range(self.parallel_config.pipeline_parallel_size)
        ]
        self.gpu_cache = [
            self.cache_engine[ve].gpu_cache
            for ve in range(self.parallel_config.pipeline_parallel_size)
        ]

    def free_cache_engine(self):
        # ensure `enforce_eager=True`
        self.cache_engine = None
        self.gpu_cache = None

    # NOTE(sgm): [VERL]: adapt from _execute_model_spmd()
    def execute_model(
        self,
        execute_model_req: ExecuteModelRequest,
        intermediate_tensors: Optional[IntermediateTensors] = None
    ) -> Optional[List[SamplerOutput]]:
        """
        Execute model in Single Program Multiple Data (SPMD) fashion.
        All workers take the same request, prepare the input and
        execute the model.
        """
        assert execute_model_req is not None, (
            "_execute_model_spmd() requires each worker to take in an "
            "ExecuteModelRequest")
        worker_input: WorkerInput = self.prepare_worker_input(
            execute_model_req=execute_model_req)
        model_input: ModelRunnerInputBase = (
            self.model_runner.prepare_model_input(
                execute_model_req.seq_group_metadata_list))

        self.execute_worker(worker_input)

        # If there is no input, we don't need to execute the model.
        if worker_input.num_seq_groups == 0:
            return []

        kwargs = extract_previous_hidden_states(execute_model_req)

        return self.model_runner.execute_model(
            model_input=model_input,
            kv_caches=self.kv_cache[worker_input.virtual_engine]
            if self.kv_cache is not None else None,
            intermediate_tensors=intermediate_tensors,
            **kwargs,
        )

    # assume the input is .state_dict()
    def sync_model_weights(self, actor_weights: Dict, load_format: str):
        if load_format in [LoadFormat.MEGATRON, LoadFormat.AUTO]:
            load_megatron_weights(actor_weights, self.model_runner.model)
        elif load_format == LoadFormat.HF:
            # full model state dict without no sharding
            load_hf_weights(actor_weights, self.model_runner.model)
        elif load_format == LoadFormat.DTENSOR:
            load_dtensor_weights(actor_weights, self.model_runner.model)

    def offload_model_weights(self) -> None:
        if self.cpu_model == None:
            self.cpu_model = {}
            for name, params in self.model_runner.model.named_parameters():
                self.cpu_model[name] = torch.empty_like(params, device='cpu')
                params.data = self.cpu_model[name]
        else:
            for name, params in self.model_runner.model.named_parameters():
                params.data = self.cpu_model[name]

def init_worker_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    # NOTE(sgm) use tcp://localhost:xxxx will hang in HF setting without megatron
    init_distributed_environment(parallel_config.world_size, rank,
                                 distributed_init_method, local_rank)

    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)
    # TODO(sgm): check whether need this
    # if pynccl_utils.is_initialized():
    #     pynccl_world_size = pynccl_utils.get_world_size()
    #     if pynccl_world_size != parallel_config.world_size:
    #         raise RuntimeError(
    #             "pynccl is already initialized but the pynccl world "
    #             "size does not match parallel_config.world_size "
    #             f"({pynccl_world_size} vs. {parallel_config.world_size}).")
    # elif parallel_config.world_size > 1:
    #     # NOTE(woosuk): We don't initialize pynccl process group when world size
    #     # is 1.
    #     # NOTE(kaichao): By default, pynccl is initialized for tp group.
    #     pynccl_utils.init_process_group(
    #         group=get_tensor_model_parallel_cpu_group())

    # # Initialize a custom fast all-reduce implementation.
    # if not parallel_config.disable_custom_all_reduce:
    #     init_custom_ar()
    # A small all_reduce for warmup.
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    # if pynccl_utils.is_initialized():
    #     pynccl_utils.all_reduce(torch.zeros(1).cuda())

def extract_previous_hidden_states(
        data: Union[ExecuteModelRequest, Dict[str, torch.Tensor]]) -> \
            Dict[str, torch.Tensor]:
    """If data contains previous_hidden_states, extract it. This returns a dict
    which can be used directly as additional kwargs in any following 
    execute_model calls. This is used in draft models like EAGLE."""
    output = {}

    # When called from non-driver worker, data is dict but when called from
    # driver worker, data is ExecuteModelRequest.
    if isinstance(data, dict):
        if "previous_hidden_states" in data:
            output["previous_hidden_states"] = data["previous_hidden_states"]
    elif data.previous_hidden_states is not None:
        output["previous_hidden_states"] = data.previous_hidden_states\
            .hidden_states

    return output