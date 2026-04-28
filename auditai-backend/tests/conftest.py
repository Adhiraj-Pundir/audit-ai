"""
Stub out ML dependencies so smoke tests run without torch/transformers installed.
The actual inference functions are patched in each test via the no_model_load fixture.
"""
import sys
from types import ModuleType
from unittest.mock import MagicMock


def _make_mock_module(name: str) -> ModuleType:
    mod = ModuleType(name)
    mod.__spec__ = MagicMock()
    return mod


_STUB_MODULES = [
    "torch",
    "torch.nn",
    "transformers",
    "transformers.AutoTokenizer",
    "transformers.AutoModelForCausalLM",
    "accelerate",
    "peft",
]

for _mod_name in _STUB_MODULES:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

# torch.bfloat16 needs to be a real attribute on the mock
import torch as _torch_mock  # noqa: E402
_torch_mock.bfloat16 = "bfloat16"
_torch_mock.no_grad = MagicMock(return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False)))
