import re
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List

from copy import deepcopy

from transformers.utils import is_accelerate_available, is_bitsandbytes_available

import bitsandbytes as bnb


if is_bitsandbytes_available():
    import torch
    import torch.nn as nn

    import bitsandbytes as bnb

if is_accelerate_available():
    from accelerate import init_empty_weights
    from accelerate.utils import find_tied_parameters


class LoraLinear(bnb.nn.Linear8bitLt):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        **kwargs
    ):
        super().__init__(self, in_features, out_features, **kwargs)
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout)

        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor):
        result = super().forward(x)
        if self.r > 0:
            lora_result = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            result += lora_result
        return result


class Linear8bitLt(nn.Linear):
    def __init__(
        self,
        input_features,
        output_features,
        bias=True,
        has_fp16_weights=True,
        memory_efficient_backward=False,
        threshold=0.0,
        index=None,
        lora_dim=0,
        lora_alpha=0,
        lora_dropout=0
    ):
        super().__init__(
            input_features, output_features, bias
        )
        self.state = bnb.MatmulLtState()
        self.index = index

        self.state.threshold = threshold
        self.state.has_fp16_weights = has_fp16_weights
        self.state.memory_efficient_backward = memory_efficient_backward
        if threshold > 0.0 and not has_fp16_weights:
            self.state.use_pool = True

        self.weight = bnb.nn.Int8Params(
            self.weight.data, has_fp16_weights=has_fp16_weights, requires_grad=has_fp16_weights
        )

        self.lora_dim = lora_dim

        if self.lora_dim > 0:
            self.lora_dim = lora_dim
            self.lora_alpha = lora_alpha
            self.lora_dropout = nn.Dropout(p=lora_dropout)
            self.lora_A = nn.Parameter(self.weight.new_zeros((lora_dim, input_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((output_features, lora_dim)))
            self.scaling = self.lora_alpha / self.lora_dim
            # Freezing the pre-trained weight matrix
            # self.weight.requires_grad = False
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)


    def init_8bit_state(self):
        self.state.CB = self.weight.CB
        self.state.SCB = self.weight.SCB
        self.weight.CB = None
        self.weight.SCB = None

    def forward(self, x):
        self.state.is_training = self.training

        if self.weight.CB is not None:
            self.init_8bit_state()

        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != torch.float16:
            self.bias.data = self.bias.data.half()

        out = bnb.matmul(x, self.weight, bias=self.bias, state=self.state)
        if self.lora_dim > 0:
            lora_result = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            print(lora_result)
            out += lora_result

        if not self.state.has_fp16_weights:
            if not self.state.memory_efficient_backward and self.state.CB is not None:
                # we converted 8-bit row major to turing/ampere format in the first inference pass
                # we no longer need the row-major weight
                del self.state.CB
                self.weight.data = self.state.CxB
            elif self.state.memory_efficient_backward and self.state.CxB is not None:
                # For memory efficient backward, we convert 8-bit row major to turing/ampere format at each inference pass.
                # Thus, we delete CxB from the state.
                del self.state.CxB

        return out


def set_module_8bit_tensor_to_device(module, tensor_name, device, value=None):
    """
    A helper function to set a given tensor (parameter of buffer) of a module on a specific device (note that doing
    `param.to(device)` creates a new tensor not linked to the parameter, which is why we need this function). The
    function is adapted from `set_module_tensor_to_device` function from accelerate that is adapted to support the
    class `Int8Params` from `bitsandbytes`.

    Args:
        module (`torch.nn.Module`):
            The module in which the tensor we want to move lives.
        tensor_name (`str`):
            The full name of the parameter/buffer.
        device (`int`, `str` or `torch.device`):
            The device on which to set the tensor.
        value (`torch.Tensor`, *optional*):
            The value of the tensor (useful when going from the meta device to any other device).
    """
    # Recurse if needed
    if "." in tensor_name:
        splits = tensor_name.split(".")
        for split in splits[:-1]:
            new_module = getattr(module, split)
            if new_module is None:
                raise ValueError(f"{module} has no attribute {split}.")
            module = new_module
        tensor_name = splits[-1]

    if tensor_name not in module._parameters and tensor_name not in module._buffers:
        raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")
    is_buffer = tensor_name in module._buffers
    old_value = getattr(module, tensor_name)

    if old_value.device == torch.device("meta") and device not in ["meta", torch.device("meta")] and value is None:
        raise ValueError(f"{tensor_name} is on the meta device, we need a `value` to put in on {device}.")

    if is_buffer:
        has_fp16_weights = None
    else:
        has_fp16_weights = getattr(module._parameters[tensor_name], "has_fp16_weights", None)

    if has_fp16_weights is not None:
        param = module._parameters[tensor_name]
        if param.device.type != "cuda":
            if value is None:
                new_value = old_value.to(device)
            elif isinstance(value, torch.Tensor):
                new_value = value.to("cpu")
                if value.dtype == torch.int8:
                    raise ValueError(
                        "You cannot load weights that are saved in int8 using `load_in_8bit=True`, make sure you are",
                        " using `load_in_8bit=True` on float32/float16/bfloat16 weights.",
                    )
            else:
                new_value = torch.tensor(value, device="cpu")
            new_value = bnb.nn.Int8Params(new_value, requires_grad=False, has_fp16_weights=has_fp16_weights).to(device)
            module._parameters[tensor_name] = new_value
    else:
        if value is None:
            new_value = old_value.to(device)
        elif isinstance(value, torch.Tensor):
            new_value = value.to(device)
        else:
            new_value = torch.tensor(value, device=device)

        if is_buffer:
            module._buffers[tensor_name] = new_value
        else:
            new_value = nn.Parameter(new_value, requires_grad=old_value.requires_grad)
            module._parameters[tensor_name] = new_value


def replace_8bit_linear(model, threshold=6.0, modules_to_not_convert="lm_head", lora_modules_to_convert="-1", lora_dim=0, lora_alpha=0, lora_dropout=0):
    """
    A helper function to replace all `torch.nn.Linear` modules by `bnb.nn.Linear8bit` modules from the `bitsandbytes`
    library. This will enable running your models using mixed int8 precision as described by the paper `GPT3.int8():
    8-bit Matrix Multiplication for Transformers at Scale`. Make sure `bitsandbytes` compiled with the correct CUDA
    version of your hardware is installed before running this function. `pip install -i https://test.pypi.org/simple/
    bitsandbytes`

    The function will be run recursively and replace all `torch.nn.Linear` modules except for the `lm_head` that should
    be kept as a `torch.nn.Linear` module. The replacement is done under `init_empty_weights` context manager so no
    CPU/GPU memory is required to run this function. Int8 mixed-precision matrix decomposition works by separating a
    matrix multiplication into two streams: (1) and systematic feature outlier stream matrix multiplied in fp16
    (0.01%), (2) a regular stream of int8 matrix multiplication (99.9%). With this method, int8 inference with no
    predictive degradation is possible for very large models (>=176B parameters).

    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        threshold (`float`, *optional*, defaults to 6.0):
            `int8_threshold` for outlier detection as described in the formentioned paper. This parameters is set to
            `6.0` as described by the paper.
        modules_to_not_convert (`str`, *optional*, defaults to `lm_head`):
            Name of the module to not convert in `Linear8bitLt`. In practice we keep the `lm_head` in full precision
            for numerical stability reasons.
    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_8bit_linear(module, threshold, modules_to_not_convert, lora_modules_to_convert, lora_dim, lora_alpha, lora_dropout)

        if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
            with init_empty_weights():
                model._modules[name] = Linear8bitLt(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    has_fp16_weights=False,
                    threshold=threshold,
                    lora_dim=0 if re.match(lora_modules_to_convert, name) else lora_dim,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout
                )
    return model


def get_keys_to_not_convert(model):
    r"""
    An utility function to get the key of the module to keep in full precision if any For example for CausalLM modules
    we may want to keep the lm_head in full precision for numerical stability reasons. For other architectures, we want
    to keep the tied weights of the model. The function will return a list of the keys of the modules to not convert in
    int8.

    Parameters:
    model (`torch.nn.Module`):
        Input model
    """
    # Create a copy of the model and tie the weights, then
    # check if it contains tied weights
    tied_model = deepcopy(model)  # this has 0 cost since it is done inside `init_empty_weights` context manager`
    tied_model.tie_weights()

    tied_keys = list(find_tied_parameters(tied_model).values())
    has_tied_params = len(tied_keys) > 0

    # Check if it is a base model
    is_base_model = not hasattr(model, model.base_model_prefix)

    # Ignore this for base models (BertModel, GPT2Model, etc.)
    if (not has_tied_params) and is_base_model:
        return []

    # otherwise they have an attached head
    list_modules = list(model.named_parameters())
    list_last_module = [list_modules[-1][0]]

    # add last module together with tied weights
    intersection = set(list_last_module) - set(tied_keys)
    list_untouched = tied_keys + list(intersection)

    return [module_name.split(".")[0] for module_name in list_untouched]
