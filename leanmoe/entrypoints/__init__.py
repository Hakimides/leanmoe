"""离线入口包。"""

from leanmoe.entrypoints.llm import LLM, load_state_dict_from_dir

__all__ = ["LLM", "load_state_dict_from_dir"]
