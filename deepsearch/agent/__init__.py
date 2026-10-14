"""Agent components for the DeepSearch LangGraph workflow."""

from .state import AgentState, empty_state

_LAZY_EXPORTS = {
    "ReflectionErrorType": ("deepsearch.agent.reflection_parser", "ReflectionErrorType"),
    "ReflectionParseError": ("deepsearch.agent.reflection_parser", "ReflectionParseError"),
    "ReflectionParseResult": ("deepsearch.agent.reflection_parser", "ReflectionParseResult"),
    "build_repair_prompt": ("deepsearch.agent.reflection_parser", "build_repair_prompt"),
    "extract_json_from_content": (
        "deepsearch.agent.reflection_parser",
        "extract_json_from_content",
    ),
    "parse_reflection_json": ("deepsearch.agent.reflection_parser", "parse_reflection_json"),
}


def __getattr__(name: str) -> object:
    """Load heavier agent exports only when callers request them."""

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "AgentState",
    "ReflectionErrorType",
    "ReflectionParseError",
    "ReflectionParseResult",
    "build_repair_prompt",
    "empty_state",
    "extract_json_from_content",
    "parse_reflection_json",
]
