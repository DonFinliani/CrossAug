import logging
import os
import time

import dotenv
import tiktoken
from openai import OpenAI

from .base_language_model import BaseLanguageModel

logger = logging.getLogger(__name__)
# Disable OpenAI and httpx logging
# Configure logging level for specific loggers by name
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

dotenv.load_dotenv()

os.environ["TIKTOKEN_CACHE_DIR"] = "./tmp"

OPENAI_MODEL = ["gpt-4", "gpt-3.5-turbo"]
DEFAULT_TOKEN_LIMIT = 32768


def _resolve_base_url(base_url: str | None = None) -> str | None:
    resolved_base_url = (
        base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    )
    if resolved_base_url and "://" not in resolved_base_url:
        resolved_base_url = f"http://{resolved_base_url}"
    return resolved_base_url


def _approximate_token_len(text: str) -> int:
    return max(1, len(text.encode("utf-8")) // 4)


def get_token_limit(model: str = "gpt-4", default_token_limit: int | None = None) -> int:
    """Returns the token limitation of provided model"""
    if model in ["gpt-4", "gpt-4-0613"]:
        num_tokens_limit = 8192
    elif model in ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]:
        num_tokens_limit = 128000
    elif model in ["gpt-3.5-turbo-16k", "gpt-3.5-turbo-16k-0613"]:
        num_tokens_limit = 16384
    elif model in [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0613",
        "text-davinci-003",
        "text-davinci-002",
    ]:
        num_tokens_limit = 4096
    else:
        env_token_limit = os.getenv("OPENAI_MAX_CONTEXT_TOKENS")
        if default_token_limit is not None:
            num_tokens_limit = default_token_limit
        elif env_token_limit:
            num_tokens_limit = int(env_token_limit)
        else:
            logger.warning(
                "Token limit is not configured for model %s. Using default limit %s. "
                "Set OPENAI_MAX_CONTEXT_TOKENS or llm.maximun_token to override.",
                model,
                DEFAULT_TOKEN_LIMIT,
            )
            num_tokens_limit = DEFAULT_TOKEN_LIMIT
    return num_tokens_limit


class ChatGPT(BaseLanguageModel):
    """A class that interacts with OpenAI's ChatGPT models through their API.

    This class provides functionality to generate text using ChatGPT models while handling
    token limits, retries, and various input formats.

    Args:
        model_name_or_path (str): The name or path of the ChatGPT model to use
        retry (int, optional): Number of retries for failed API calls. Defaults to 5

    Attributes:
        retry (int): Maximum number of retry attempts for failed API calls
        model_name (str): Name of the ChatGPT model being used
        maximun_token (int): Maximum token limit for the specified model
        client (OpenAI): OpenAI client instance for API interactions

    Methods:
        token_len(text): Calculate the number of tokens in a given text
        generate_sentence(llm_input, system_input): Generate response using the ChatGPT model

    Raises:
        Exception: If generation fails after maximum retries
    """

    def __init__(
        self,
        model_name_or_path: str,
        retry: int = 1,
        base_url: str | None = None,
        api_key: str | None = None,
        maximun_token: int | None = None,
        maximum_token: int | None = None,
    ):
        self.retry = retry
        self.model_name = model_name_or_path
        token_limit = maximum_token if maximum_token is not None else maximun_token
        self.maximun_token = get_token_limit(self.model_name, token_limit)
        self.base_url = _resolve_base_url(base_url)
        self._token_encoding = None
        self._use_approx_token_len = False

        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if self.base_url and not resolved_api_key:
            resolved_api_key = "EMPTY"

        client_kwargs = {"api_key": resolved_api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        client = OpenAI(**client_kwargs)
        self.client = client

    def token_len(self, text: str) -> int:
        """Returns the number of tokens used by a list of messages."""
        if self._use_approx_token_len:
            return _approximate_token_len(text)
        if self._token_encoding is not None:
            return len(self._token_encoding.encode(text))

        try:
            self._token_encoding = tiktoken.encoding_for_model(self.model_name)
            return len(self._token_encoding.encode(text))
        except KeyError:
            try:
                self._token_encoding = tiktoken.get_encoding("cl100k_base")
                return len(self._token_encoding.encode(text))
            except Exception as e:
                self._use_approx_token_len = True
                logger.warning(
                    "Could not load tiktoken fallback encoding for model %s: %s. "
                    "Using approximate token length.",
                    self.model_name,
                    e,
                )
                return _approximate_token_len(text)
        except Exception as e:
            self._use_approx_token_len = True
            logger.warning(
                "Could not count tokens with tiktoken for model %s: %s. "
                "Using approximate token length.",
                self.model_name,
                e,
            )
            return _approximate_token_len(text)

    def generate_sentence(
        self, llm_input: str | list, system_input: str = ""
    ) -> str | Exception:
        """Generate a response using the ChatGPT API.

        This method sends a request to the ChatGPT API and returns the generated response.
        It handles both single string inputs and message lists, with retry logic for failed attempts.

        Args:
            llm_input (Union[str, list]): Either a string containing the user's input or a list of message dictionaries
                in the format [{"role": "role_type", "content": "message_content"}, ...]
            system_input (str, optional): System message to be prepended to the conversation. Defaults to "".

        Returns:
            Union[str, Exception]: The generated response text if successful, or the Exception if all retries fail.
                The response is stripped of leading/trailing whitespace.

        Raises:
            Exception: If all retry attempts fail, returns the last encountered exception.

        Notes:
            - Automatically truncates inputs that exceed the maximum token limit
            - Uses exponential backoff with 30 second delays between retries
            - Sets temperature to 0.0 for deterministic outputs
            - Timeout is set to 60 seconds per API call
        """

        # If the input is a list, it is assumed that the input is a list of messages
        if isinstance(llm_input, list):
            message = llm_input
        else:
            message = []
            if system_input:
                message.append({"role": "system", "content": system_input})
            message.append({"role": "user", "content": llm_input})
        cur_retry = 0
        num_retry = self.retry
        # Check if the input is too long
        message_string = "\n".join([m["content"] for m in message])
        input_length = self.token_len(message_string)
        if input_length > self.maximun_token:
            print(
                f"Input lengt {input_length} is too long. The maximum token is {self.maximun_token}.\n Right tuncate the input to {self.maximun_token} tokens."
            )
            llm_input = llm_input[: self.maximun_token]
        error = Exception("Failed to generate sentence")
        while cur_retry <= num_retry:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name, messages=message, timeout=120, temperature=0.0
                )
                result = response.choices[0].message.content.strip()  # type: ignore
                return result
            except Exception as e:
                logger.error("Message: %s", llm_input)
                logger.error("Number of token: %s", self.token_len(message_string))
                logger.error(e)
                time.sleep(30)
                cur_retry += 1
                error = e
                continue
        return error
