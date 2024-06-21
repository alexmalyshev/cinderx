# pyre-unsafe
from types import ModuleType
from typing import Any, List, Mapping, Type

class async_cached_classproperty:
    func: Any
    name: Any
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...
    def __get__(self, instance, owner) -> Any: ...

class async_cached_property:
    fget: Any
    func: Any
    name: Any
    slot: Any
    def __init__(self, *args, **kwargs) -> None: ...
    def __get__(self, instance, owner) -> Any: ...

class cached_classproperty:
    func: Any
    name: Any
    __name__: Any
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...
    def __get__(self, instance, owner) -> Any: ...

class cached_property:
    fget: Any
    func: Any
    name: Any
    slot: Any
    __name__: Any
    def __init__(self, *args, **kwargs) -> None: ...
    def clear(self, *args, **kwargs) -> Any: ...
    def has_value(self, *args, **kwargs) -> Any: ...
    def __get__(self, instance, owner) -> Any: ...
    def __set_name__(self, *args, **kwargs) -> Any: ...

class _PatchEnabledDescr:
    def __get__(self, inst: StrictModule, typ: Type[StrictModule]) -> bool: ...

class StrictModule(ModuleType):
    __name__: Any
    __patch_enabled__: _PatchEnabledDescr
    def __init__(self, d: Mapping[str, object], enable_patching: bool) -> None: ...
    def patch(self, name: str, value: object) -> None: ...
    def patch_delete(self, name: str) -> None: ...
    def __delattr__(self, name) -> Any: ...
    def __dir__(self) -> List: ...
    def __setattr__(self, name, value) -> Any: ...

def init() -> None: ...
def _compile_perf_trampoline_pre_fork(*args, **kwargs) -> Any: ...
def _get_entire_call_stack_as_qualnames_with_lineno_and_frame(
    *args, **kwargs
) -> Any: ...
def _get_entire_call_stack_as_qualnames_with_lineno(*args, **kwargs) -> Any: ...
def _is_compile_perf_trampoline_pre_fork_enabled(*args, **kwargs) -> Any: ...
def clear_all_shadow_caches(*args, **kwargs) -> Any: ...
def clear_caches(*args, **kwargs) -> Any: ...
def clear_classloader_caches(*args, **kwargs) -> Any: ...
def clear_type_profiles(*args, **kwargs) -> Any: ...
def disable_parallel_gc(*args, **kwargs) -> Any: ...
def enable_parallel_gc(*args, **kwargs) -> Any: ...
def get_and_clear_type_profiles_with_metadata(*args, **kwargs) -> Any: ...
def get_and_clear_type_profiles(*args, **kwargs) -> Any: ...
def get_parallel_gc_settings(*args, **kwargs) -> Any: ...
def set_profile_interp_all(*args, **kwargs) -> Any: ...
def set_profile_interp_period(*args, **kwargs) -> Any: ...
def set_profile_interp(*args, **kwargs) -> Any: ...
def strict_module_patch_delete(mod, name) -> Any: ...
def strict_module_patch_enabled(mod) -> Any: ...
def strict_module_patch(mod, name, value) -> Any: ...
def watch_sys_modules(): ...

class StaticTypeError(TypeError):
    ...
