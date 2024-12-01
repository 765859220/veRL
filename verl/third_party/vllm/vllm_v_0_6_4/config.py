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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/config.py

import enum
import json
import warnings
from dataclasses import dataclass, field, replace
from typing import (Any, Callable, ClassVar, Counter, Dict,
                    Final, List, Literal, Mapping, Optional, Set, Tuple, Type,
                    Union)

import torch
from transformers import PretrainedConfig

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.transformers_utils.config import (
    ConfigFormat, get_hf_image_processor_config,
    get_hf_text_config)
from vllm.utils import (identity, print_warning_once)

# Add for verl
from vllm.config import ModelConfig, _get_and_verify_dtype, _get_and_verify_max_len, get_served_model_name

logger = init_logger(__name__)


TaskOption = Literal["auto", "generate", "embedding"]

# "draft" is only used internally for speculative decoding
_Task = Literal["generate", "embedding", "draft"]

HfOverrides = Union[Dict[str, Any], Callable[[PretrainedConfig],
                                             PretrainedConfig]]


class ModelConfig(ModelConfig):
    """Configuration for the model.

    Args:
        model: Name or path of the huggingface model to use.
            It is also used as the content for `model_name` tag in metrics
            output when `served_model_name` is not specified.
        task: The task to use the model for. Each vLLM instance only supports
            one task, even if the same model can be used for multiple tasks.
            When the model only supports one task, "auto" can be used to select
            it; otherwise, you must specify explicitly which task to use.
        tokenizer: Name or path of the huggingface tokenizer to use.
        tokenizer_mode: Tokenizer mode. "auto" will use the fast tokenizer if
            available, "slow" will always use the slow tokenizer, and
            "mistral" will always use the tokenizer from `mistral_common`.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        allowed_local_media_path: Allowing API requests to read local images or
            videos from directories specified by the server file system.
            This is a security risk. Should only be enabled in trusted
            environments.
        dtype: Data type for model weights and activations. The "auto" option
            will use FP16 precision for FP32 and FP16 models, and BF16 precision
            for BF16 models.
        seed: Random seed for reproducibility.
        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id. If unspecified, will use the default
            version.
        code_revision: The specific revision to use for the model code on
            Hugging Face Hub. It can be a branch name, a tag name, or a
            commit id. If unspecified, will use the default version.
        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id. If unspecified, will use
            the default version.
        max_model_len: Maximum length of a sequence (including prompt and
            output). If None, will be derived from the model.
        quantization: Quantization method that was used to quantize the model
            weights. If None, we assume the model weights are not quantized.
        quantization_param_path: Path to JSON file containing scaling factors.
            Used to load KV cache scaling factors into the model when KV cache
            type is FP8_E4M3 on ROCm (AMD GPU). In the future these will also
            be used to load activation and weight scaling factors when the
            model dtype is FP8_E4M3 on ROCm.
        enforce_eager: Whether to enforce eager execution. If True, we will
            disable CUDA graph and always execute the model in eager mode.
            If False, we will use CUDA graph and eager execution in hybrid.
            If None, the user did not specify, so default to False.
        max_seq_len_to_capture: Maximum sequence len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode. Additionally for encoder-decoder models, if the
            sequence length of the encoder input is larger than this, we fall
            back to the eager mode.
        disable_sliding_window: Whether to disable sliding window. If True,
            we will disable the sliding window functionality of the model.
            If the model does not support sliding window, this argument is
            ignored.
        skip_tokenizer_init: If true, skip initialization of tokenizer and
            detokenizer.
        served_model_name: The model name used in metrics tag `model_name`,
            matches the model name exposed via the APIs. If multiple model
            names provided, the first name will be used. If not specified,
            the model name will be the same as `model`.
        limit_mm_per_prompt: Maximum number of data items per modality
            per prompt. Only applicable for multimodal models.
        config_format: The config format which shall be loaded.
            Defaults to 'auto' which defaults to 'hf'.
        hf_overrides: If a dictionary, contains arguments to be forwarded to the
            HuggingFace config. If a callable, it is called to update the
            HuggingFace config.
        mm_processor_kwargs: Arguments to be forwarded to the model's processor
            for multi-modal data, e.g., image processor.
        override_neuron_config: Initialize non default neuron config or
            override default neuron config that are specific to Neuron devices,
            this argument will be used to configure the neuron config that
            can not be gathered from the vllm arguments.
        override_pooling_config: Initialize non default pooling config or
            override default pooling config for the embedding model.
    """

    def __init__(
            self,
            hf_config: PretrainedConfig,
            task: Union[TaskOption, _Task],
            tokenizer_mode: str,
            trust_remote_code: bool,
            dtype: Union[str, torch.dtype],
            seed: int,
            allowed_local_media_path: str = "",
            revision: Optional[str] = None,
            code_revision: Optional[str] = None,
            rope_scaling: Optional[Dict[str, Any]] = None,
            rope_theta: Optional[float] = None,
            tokenizer_revision: Optional[str] = None,
            max_model_len: Optional[int] = None,
            spec_target_max_model_len: Optional[int] = None,
            quantization: Optional[str] = None,
            quantization_param_path: Optional[str] = None,
            enforce_eager: Optional[bool] = None,
            max_seq_len_to_capture: Optional[int] = None,
            max_logprobs: int = 20,
            disable_sliding_window: bool = False,
            skip_tokenizer_init: bool = False,
            served_model_name: Optional[Union[str, List[str]]] = None,
            limit_mm_per_prompt: Optional[Mapping[str, int]] = None,
            use_async_output_proc: bool = True,
            config_format: ConfigFormat = ConfigFormat.AUTO,
            hf_overrides: Optional[HfOverrides] = None,
            mm_processor_kwargs: Optional[Dict[str, Any]] = None,
            override_neuron_config: Optional[Dict[str, Any]] = None,
            override_pooler_config: Optional["PoolerConfig"] = None) -> None:
        self.model = hf_config._name_or_path
        self.tokenizer = hf_config._name_or_path
        # NOTE(sgm): same as open-sourced
        self.tokenizer_mode = tokenizer_mode
        self.trust_remote_code = trust_remote_code
        self.allowed_local_media_path = allowed_local_media_path
        self.seed = seed
        self.revision = revision
        self.code_revision = code_revision

        if hf_overrides is None:
            hf_overrides = {}

        if callable(hf_overrides):
            hf_overrides_kw = {}
            hf_overrides_fn = hf_overrides
        else:
            hf_overrides_kw = hf_overrides
            hf_overrides_fn = identity

        if rope_scaling is not None:
            hf_override: Dict[str, Any] = {"rope_scaling": rope_scaling}
            hf_overrides_kw.update(hf_override)
            msg = ("`--rope-scaling` will be removed in a future release. "
                   f"'Please instead use `--hf-overrides '{hf_override!r}'`")
            warnings.warn(DeprecationWarning(msg), stacklevel=2)
        if rope_theta is not None:
            hf_override = {"rope_theta": rope_theta}
            hf_overrides_kw.update(hf_override)
            msg = ("`--rope-theta` will be removed in a future release. "
                   f"'Please instead use `--hf-overrides '{hf_override!r}'`")
            warnings.warn(DeprecationWarning(msg), stacklevel=2)

        # The tokenizer version is consistent with the model version by default.
        if tokenizer_revision is None:
            self.tokenizer_revision = revision
        else:
            self.tokenizer_revision = tokenizer_revision
        self.quantization = quantization
        self.quantization_param_path = quantization_param_path
        self.enforce_eager = enforce_eager
        self.max_seq_len_to_capture = max_seq_len_to_capture
        self.max_logprobs = max_logprobs
        self.disable_sliding_window = disable_sliding_window
        self.skip_tokenizer_init = skip_tokenizer_init

        # hf_config = get_config(self.model, trust_remote_code, revision,
        #                        code_revision, config_format, **hf_overrides_kw)
        hf_config = hf_overrides_fn(hf_config)
        self.hf_config = hf_config

        self.hf_text_config = get_hf_text_config(self.hf_config)
        self.encoder_config = self._get_encoder_config()
        self.hf_image_processor_config = get_hf_image_processor_config(
            self.model, revision)
        self.dtype = _get_and_verify_dtype(self.hf_text_config, dtype)
        self.use_async_output_proc = use_async_output_proc
        self.mm_processor_kwargs = mm_processor_kwargs

        # Set enforce_eager to False if the value is unset.
        if self.enforce_eager is None:
            self.enforce_eager = False

        sliding_window = getattr(self.hf_text_config, "sliding_window", None)
        has_interleaved_attention = (sliding_window is not None) and (
            isinstance(sliding_window, list) or
            (self.hf_text_config.model_type in ["gemma2"]))

        if (not self.disable_sliding_window and has_interleaved_attention):
            sliding_window_len_min = get_min_sliding_window(
                self.hf_text_config.sliding_window)

            print_warning_once(
                f"{self.hf_text_config.model_type} has interleaved attention, "
                "which is currently not supported by vLLM. Disabling sliding "
                "window and capping the max length to the sliding window size "
                f"({sliding_window_len_min}).")
            self.disable_sliding_window = True

        self.max_model_len = _get_and_verify_max_len(
            hf_config=self.hf_text_config,
            max_model_len=max_model_len,
            disable_sliding_window=self.disable_sliding_window,
            sliding_window_len=self.get_hf_config_sliding_window(),
            spec_target_max_model_len=spec_target_max_model_len,
            encoder_config=self.encoder_config)
        self.served_model_name = get_served_model_name(self.model,
                                                       served_model_name)
        self.multimodal_config = self._init_multimodal_config(
            limit_mm_per_prompt)
        if not self.skip_tokenizer_init:
            self._verify_tokenizer_mode()

        self.is_attention_free = self._init_attention_free()
        self.has_inner_state = self._init_has_inner_state()

        if current_platform.is_neuron():
            self.override_neuron_config = override_neuron_config
        else:
            self.override_neuron_config = None

        supported_tasks, task = self._resolve_task(task, self.hf_config)
        self.supported_tasks = supported_tasks
        self.task: Final = task
        self.pooler_config = self._init_pooler_config(override_pooler_config)

        self._verify_quantization()
        self._verify_cuda_graph()
        self._verify_bnb_config()

class LoadFormat(str, enum.Enum):
    AUTO = 'auto'
    MEGATRON = "megatron"
    HF = "hf"
    DTENSOR = 'dtensor'
    DUMMY_HF = 'dummy_hf'
    DUMMY_MEGATRON = 'dummy_megatron'
    DUMMY_DTENSOR = 'dummy_dtensor'

@dataclass
class LoadConfig:
    """
        download_dir: Directory to download and load the weights, default to the
            default cache directory of huggingface.
        load_format: The format of the model weights to load:
            "auto" will try to load the weights in the safetensors format and
                fall back to the pytorch bin format if safetensors format is
                not available.
            "pt" will load the weights in the pytorch bin format.
            "safetensors" will load the weights in the safetensors format.
            "npcache" will load the weights in pytorch format and store
                a numpy cache to speed up the loading.
            "dummy" will initialize the weights with random values, which is
                mainly for profiling.
            "tensorizer" will use CoreWeave's tensorizer library for
                fast weight loading.
            "bitsandbytes" will load nf4 type weights.
        ignore_patterns: The list of patterns to ignore when loading the model.
            Default to "original/**/*" to avoid repeated loading of llama's
            checkpoints.
    """

    load_format: Union[str, LoadFormat, "BaseModelLoader"] = LoadFormat.AUTO
    download_dir: Optional[str] = None
    model_loader_extra_config: Optional[Union[str, dict]] = field(
        default_factory=dict)
    ignore_patterns: Optional[Union[List[str], str]] = None

    def __post_init__(self):
        model_loader_extra_config = self.model_loader_extra_config or {}
        if isinstance(model_loader_extra_config, str):
            self.model_loader_extra_config = json.loads(
                model_loader_extra_config)
        self._verify_load_format()

        if self.ignore_patterns is not None and len(self.ignore_patterns) > 0:
            logger.info(
                "Ignoring the following patterns when downloading weights: %s",
                self.ignore_patterns)
        else:
            self.ignore_patterns = ["original/**/*"]

    def _verify_load_format(self) -> None:
        if not isinstance(self.load_format, str):
            return

        load_format = self.load_format.lower()
        self.load_format = LoadFormat(load_format)

        rocm_not_supported_load_format: List[str] = []
        if current_platform.is_rocm(
        ) and load_format in rocm_not_supported_load_format:
            rocm_supported_load_format = [
                f for f in LoadFormat.__members__
                if (f not in rocm_not_supported_load_format)
            ]
            raise ValueError(
                f"load format '{load_format}' is not supported in ROCm. "
                f"Supported load formats are "
                f"{rocm_supported_load_format}")



def get_min_sliding_window(
        sliding_window: Union[int, List[Optional[int]]]) -> int:
    if isinstance(sliding_window, list):
        return min(s for s in sliding_window if s is not None)

    return sliding_window