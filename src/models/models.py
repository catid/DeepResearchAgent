import os
from openai import OpenAI, AsyncOpenAI
from typing import Dict, Any, Tuple

from dotenv import load_dotenv
load_dotenv(verbose=True)

from langchain_openai import ChatOpenAI

from src.logger import logger
from src.models.litellm import LiteLLMModel
from src.models.openaillm import OpenAIServerModel
from src.models.hfllm import InferenceClientModel
from src.models.restful import (RestfulModel,
                                RestfulTranscribeModel,
                                RestfulImagenModel,
                                RestfulVeoPridictModel,
                                RestfulVeoFetchModel,
                                RestfulResponseModel)
from src.utils import Singleton
from src.proxy.local_proxy import HTTP_CLIENT, ASYNC_HTTP_CLIENT

custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}
PLACEHOLDER = "PLACEHOLDER"


class ModelManager(metaclass=Singleton):
    def __init__(self):
        self.registed_models: Dict[str, Any] = {}
        
    def init_models(self, use_local_proxy: bool = False):
        self._register_openai_models(use_local_proxy=use_local_proxy)
        self._register_anthropic_models(use_local_proxy=use_local_proxy)
        self._register_google_models(use_local_proxy=use_local_proxy)
        self._register_qwen_models(use_local_proxy=use_local_proxy)
        self._register_langchain_models(use_local_proxy=use_local_proxy)
        self._register_vllm_models(use_local_proxy=use_local_proxy)
        self._register_deepseek_models(use_local_proxy=use_local_proxy)

    def _check_local_api_key(
        self,
        local_api_key_name: str,
        remote_api_key_name: str,
        default: str | None = None,
    ) -> str:
        api_key = os.getenv(local_api_key_name, PLACEHOLDER)
        if api_key != PLACEHOLDER:
            return api_key

        if default is not None:
            logger.warning(
                "Local API key %s is not set, falling back to default value.",
                local_api_key_name,
            )
            return default

        logger.warning(
            "Local API key %s is not set, using remote API key %s",
            local_api_key_name,
            remote_api_key_name,
        )
        return os.getenv(remote_api_key_name, PLACEHOLDER)
    
    def _check_local_api_base(
        self,
        local_api_base_name: str,
        remote_api_base_name: str,
        default: str | None = None,
    ) -> str:
        api_base = os.getenv(local_api_base_name, PLACEHOLDER)
        if api_base != PLACEHOLDER:
            return api_base

        if default is not None:
            logger.warning(
                "Local API base %s is not set, falling back to default value.",
                local_api_base_name,
            )
            return default

        logger.warning(
            "Local API base %s is not set, using remote API base %s",
            local_api_base_name,
            remote_api_base_name,
        )
        api_base = os.getenv(remote_api_base_name, PLACEHOLDER)
        if api_base == PLACEHOLDER and default is not None:
            logger.warning(
                "Remote API base %s is not set, using default value.",
                remote_api_base_name,
            )
            return default
        return api_base
    
    def _register_openai_models(self, use_local_proxy: bool = False):
        # gpt-4o, gpt-4.1, o1, o3, gpt-4o-search-preview
        if use_local_proxy:
            logger.info("Using locally hosted OpenAI-compatible models")
            api_key = self._check_local_api_key(
                local_api_key_name="LOCAL_OPENAI_API_KEY",
                remote_api_key_name="OPENAI_API_KEY",
                default="local-dev-key",
            )
            api_base = self._check_local_api_base(
                local_api_base_name="LOCAL_OPENAI_API_BASE",
                remote_api_base_name="OPENAI_API_BASE",
                default="http://ripper.lan:8000/v1",
            )

            default_chat_model_id = os.getenv("LOCAL_OPENAI_CHAT_MODEL_ID", "local-openai")
            default_reasoner_model_id = os.getenv(
                "LOCAL_OPENAI_REASONER_MODEL_ID", default_chat_model_id
            )
            default_responses_model_id = os.getenv(
                "LOCAL_OPENAI_DEEP_RESEARCH_MODEL_ID", default_reasoner_model_id
            )

            shared_async_client = AsyncOpenAI(
                api_key=api_key,
                base_url=api_base,
                http_client=ASYNC_HTTP_CLIENT,
            )

            chat_models = [
                {"model_name": "gpt-4o", "env": "LOCAL_GPT_4O_MODEL_ID"},
                {"model_name": "gpt-4.1", "env": "LOCAL_GPT_4_1_MODEL_ID"},
                {"model_name": "o1", "env": "LOCAL_O1_MODEL_ID"},
                {"model_name": "gpt-4o-search-preview", "env": "LOCAL_GPT_4O_SEARCH_MODEL_ID"},
                {"model_name": "gpt-5", "env": "LOCAL_GPT_5_MODEL_ID"},
            ]

            for chat_model in chat_models:
                model_id = os.getenv(chat_model["env"], default_chat_model_id)
                model = LiteLLMModel(
                    model_id=model_id,
                    http_client=shared_async_client,
                    custom_role_conversions=custom_role_conversions,
                )
                self.registed_models[chat_model["model_name"]] = model

            reasoner_model_id = os.getenv("LOCAL_O3_MODEL_ID", default_reasoner_model_id)
            o3_model = RestfulModel(
                api_base=api_base,
                api_type="chat/completions",
                api_key=api_key,
                model_id=reasoner_model_id,
                http_client=HTTP_CLIENT,
                custom_role_conversions=custom_role_conversions,
            )
            self.registed_models["o3"] = o3_model

            whisper_model_id = os.getenv("LOCAL_WHISPER_MODEL_ID", "whisper-1")
            whisper_model = RestfulTranscribeModel(
                api_base=api_base,
                api_key=api_key,
                api_type="whisper",
                model_id=whisper_model_id,
                http_client=HTTP_CLIENT,
                custom_role_conversions=custom_role_conversions,
            )
            self.registed_models["whisper"] = whisper_model

            deep_research_model_id = os.getenv(
                "LOCAL_O3_DEEP_RESEARCH_MODEL_ID", default_responses_model_id
            )
            deep_research_model = RestfulResponseModel(
                api_base=api_base,
                api_key=api_key,
                api_type="responses",
                model_id=deep_research_model_id,
                http_client=HTTP_CLIENT,
                custom_role_conversions=custom_role_conversions,
            )
            self.registed_models["o3-deep-research"] = deep_research_model
            
        else:
            logger.info("Using remote API for OpenAI models")
            api_key = self._check_local_api_key(local_api_key_name="OPENAI_API_KEY", 
                                                remote_api_key_name="OPENAI_API_KEY")
            api_base = self._check_local_api_base(local_api_base_name="OPENAI_API_BASE", 
                                                    remote_api_base_name="OPENAI_API_BASE")
            
            models = [
                {
                    "model_name": "gpt-4o",
                    "model_id": "gpt-4o",
                },
                {
                    "model_name": "gpt-4.1",
                    "model_id": "gpt-4.1",
                },
                {
                    "model_name": "o1",
                    "model_id": "o1",
                },
                {
                    "model_name": "o3",
                    "model_id": "o3",
                },
                {
                    "model_name": "gpt-4o-search-preview",
                    "model_id": "gpt-4o-search-preview",
                },
            ]
            
            for model in models:
                model_name = model["model_name"]
                model_id = model["model_id"]
                model = LiteLLMModel(
                    model_id=model_id,
                    api_key=api_key,
                    api_base=api_base,
                    custom_role_conversions=custom_role_conversions,
                )
                self.registed_models[model_name] = model
    
            
    def _register_anthropic_models(self, use_local_proxy: bool = False):
        # claude37-sonnet, claude37-sonnet-thinking
        if use_local_proxy:
            logger.info("Local Anthropic endpoint not configured; skipping registration.")
            return

        else:
            logger.info("Using remote API for Anthropic models")
            api_key = self._check_local_api_key(local_api_key_name="ANTHROPIC_API_KEY", 
                                                remote_api_key_name="ANTHROPIC_API_KEY")
            api_base = self._check_local_api_base(local_api_base_name="ANTHROPIC_API_BASE", 
                                                    remote_api_base_name="ANTHROPIC_API_BASE")
            
            models = [
                {
                    "model_name": "claude37-sonnet",
                    "model_id": "claude-3-7-sonnet-20250219",
                },
                {
                    "model_name": "claude37-sonnet-thinking",
                    "model_id": "claude-3-7-sonnet-20250219",
                },
            ]
            
            for model in models:
                model_name = model["model_name"]
                model_id = model["model_id"]
                model = LiteLLMModel(
                    model_id=model_id,
                    api_key=api_key,
                    api_base=api_base,
                    custom_role_conversions=custom_role_conversions,
                )
                self.registed_models[model_name] = model
            
    def _register_google_models(self, use_local_proxy: bool = False):
        if use_local_proxy:
            logger.info("Local Google model endpoint not configured; skipping registration.")
            return
        else:
            logger.info("Using remote API for Google models")
            api_key = self._check_local_api_key(local_api_key_name="GOOGLE_API_KEY", 
                                                remote_api_key_name="GOOGLE_API_KEY")
            api_base = self._check_local_api_base(local_api_base_name="GOOGLE_API_BASE", 
                                                    remote_api_base_name="GOOGLE_API_BASE")
            
            models = [
                {
                    "model_name": "gemini-2.5-pro",
                    "model_id": "gemini-2.5-pro-preview-06-05",
                },
            ]
            
            for model in models:
                model_name = model["model_name"]
                model_id = model["model_id"]
                model = LiteLLMModel(
                    model_id=model_id,
                    api_key=api_key,
                    # api_base=api_base,
                    custom_role_conversions=custom_role_conversions,
                )
                self.registed_models[model_name] = model
                
    def _register_qwen_models(self, use_local_proxy: bool = False):
        # qwen2.5-7b-instruct
        models = [
            {
                "model_name": "qwen2.5-7b-instruct",
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
            },
            {
                "model_name": "qwen2.5-14b-instruct",
                "model_id": "Qwen/Qwen2.5-14B-Instruct",
            },
            {
                "model_name": "qwen2.5-32b-instruct",
                "model_id": "Qwen/Qwen2.5-32B-Instruct",
            },
        ]
        for model in models:
            model_name = model["model_name"]
            model_id = model["model_id"]
            
            model = InferenceClientModel(
                model_id=model_id,
                custom_role_conversions=custom_role_conversions,
            )
            self.registed_models[model_name] = model

    def _register_langchain_models(self, use_local_proxy: bool = False):
        # langchain models
        models = [
            {
                "model_name": "langchain-gpt-4o",
                "model_id": "gpt-4o",
            },
            {
                "model_name": "langchain-gpt-4.1",
                "model_id": "gpt-4.1",
            },
            {
                "model_name": "langchain-o3",
                "model_id": "o3",
            },
        ]

        if use_local_proxy:
            logger.info("Using locally hosted models for LangChain integrations")
            api_key = self._check_local_api_key(local_api_key_name="LOCAL_OPENAI_API_KEY",
                                                remote_api_key_name="OPENAI_API_KEY")
            api_base = self._check_local_api_base(local_api_base_name="LOCAL_OPENAI_API_BASE",
                                                    remote_api_base_name="OPENAI_API_BASE",
                                                    default="http://ripper.lan:8000/v1")

            default_chat_model_id = os.getenv("LOCAL_OPENAI_CHAT_MODEL_ID", "local-openai")
            model_id_overrides = {
                "langchain-gpt-4o": os.getenv("LOCAL_GPT_4O_MODEL_ID", default_chat_model_id),
                "langchain-gpt-4.1": os.getenv("LOCAL_GPT_4_1_MODEL_ID", default_chat_model_id),
                "langchain-o3": os.getenv("LOCAL_O3_MODEL_ID", default_chat_model_id),
            }

            for model in models:
                model_name = model["model_name"]
                model_id = model_id_overrides.get(model_name, default_chat_model_id)

                model = ChatOpenAI(
                    model=model_id,
                    api_key=api_key,
                    base_url=api_base,
                    http_client=HTTP_CLIENT,
                    http_async_client=ASYNC_HTTP_CLIENT,
                )
                self.registed_models[model_name] = model

        else:
            logger.info("Using remote API for LangChain models")
            api_key = self._check_local_api_key(local_api_key_name="OPENAI_API_KEY",
                                                remote_api_key_name="OPENAI_API_KEY")
            api_base = self._check_local_api_base(local_api_base_name="OPENAI_API_BASE",
                                                    remote_api_base_name="OPENAI_API_BASE")

            for model in models:
                model_name = model["model_name"]
                model_id = model["model_id"]

                model = ChatOpenAI(
                    model=model_id,
                    api_key=api_key,
                    base_url=api_base,
                )
                self.registed_models[model_name] = model
    def _register_vllm_models(self, use_local_proxy: bool = False):
        # qwen or other vLLM hosted models
        api_key = self._check_local_api_key(
            local_api_key_name="LOCAL_VLLM_API_KEY",
            remote_api_key_name="QWEN_API_KEY",
            default="local-dev-key",
        )
        api_base = self._check_local_api_base(
            local_api_base_name="LOCAL_VLLM_API_BASE",
            remote_api_base_name="QWEN_API_BASE",
            default="http://ripper2.lan:8000/v1",
        )

        model_name = os.getenv("LOCAL_VLLM_MODEL_NAME", "local-vllm")
        model_id = os.getenv("LOCAL_VLLM_MODEL_ID", "Qwen")

        client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
        )
        model = OpenAIServerModel(
            model_id=model_id,
            http_client=client,
            custom_role_conversions=custom_role_conversions,
        )
        self.registed_models[model_name] = model

        vision_model_name = os.getenv("LOCAL_VLLM_VISION_MODEL_NAME")
        if vision_model_name:
            vision_model_id = os.getenv("LOCAL_VLLM_VISION_MODEL_ID", vision_model_name)
            client = AsyncOpenAI(
                api_key=self._check_local_api_key(
                    local_api_key_name="LOCAL_VLLM_VISION_API_KEY",
                    remote_api_key_name="QWEN_VL_API_KEY",
                    default=api_key,
                ),
                base_url=self._check_local_api_base(
                    local_api_base_name="LOCAL_VLLM_VISION_API_BASE",
                    remote_api_base_name="QWEN_VL_API_BASE",
                    default=api_base,
                ),
            )
            model = OpenAIServerModel(
                model_id=vision_model_id,
                http_client=client,
                custom_role_conversions=custom_role_conversions,
            )
            self.registed_models[vision_model_name] = model

    def _register_deepseek_models(self, use_local_proxy: bool = False):
        # deepseek models
        if use_local_proxy:
            logger.info("Local DeepSeek endpoint not configured; skipping registration.")
            return
        else:
            logger.warning("DeepSeek models are not supported in remote API mode.")
