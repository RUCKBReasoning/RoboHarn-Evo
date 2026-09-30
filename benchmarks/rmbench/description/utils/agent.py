from typing import List, Type, Optional
from pydantic import BaseModel, Field
import json
import os


ENDPOINT_ENV = "RMBENCH_DESCRIPTION_API_ENDPOINT"
API_KEY_ENV = "RMBENCH_DESCRIPTION_API_KEY"
MODEL_ENV = "RMBENCH_DESCRIPTION_API_MODEL"


def _configured_client():
    """Build the optional description client without a private default endpoint."""

    endpoint = os.environ.get(ENDPOINT_ENV, "").strip()
    api_key = (
        os.environ.get(API_KEY_ENV, "").strip()
        or os.environ.get("AZURE_API_KEY", "").strip()
    )
    missing = [
        name
        for name, value in ((ENDPOINT_ENV, endpoint), (API_KEY_ENV, api_key))
        if not value
    ]
    if missing:
        raise RuntimeError(
            "RMBench description generation requires explicit API configuration; "
            f"set {', '.join(missing)}"
        )

    try:
        from azure.ai.inference import ChatCompletionsClient
        from azure.core.credentials import AzureKeyCredential
    except ImportError as exc:
        raise RuntimeError(
            "RMBench description generation requires the optional Azure AI "
            "Inference client dependency"
        ) from exc
    return ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(api_key),
    )


def generate(messages: List[dict], custom_format: Type[BaseModel]) -> Optional[BaseModel]:
    client = _configured_client()
    model_name = os.environ.get(MODEL_ENV, "gpt-4o").strip() or "gpt-4o"
    strformat = custom_format.schema_json()
    messages.append({
        "role": "system",
        "content": "you shall output a json object with the following format: " + strformat,
    })
    response = client.complete(
        messages=messages,
        max_tokens=4096,
        temperature=0.8,
        top_p=1.0,
        model=model_name,
        response_format="json_object",
    )

    json_content = response.choices[0].message.content
    if json_content:
        parsed_json = json.loads(json_content)
        return (custom_format.parse_obj(parsed_json)
                if hasattr(custom_format, "parse_obj") else custom_format.model_validate(parsed_json))

    return None


if __name__ == "__main__":
    pass
