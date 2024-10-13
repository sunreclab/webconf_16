from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain_community.llms import Ollama

from yacs.config import CfgNode

config = CfgNode(new_allowed=True)
config.merge_from_file("config.yaml")
ollama = Ollama(base_url=config['url'], model=config['model_name'])


class OllamaModels(LLM):
    logger: Any

    @property
    def _llm_type(self) -> str:
        return "Ollama"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            history: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        response = ollama.invoke(prompt)
        if len(response) == 0:
            self.logger.error("Ollama error occurred")
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "max_token": None,
            "URL": None,
            "headers": None,
            "payload": None,
        }
