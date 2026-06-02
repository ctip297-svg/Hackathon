import os
import traceback
from typing import Dict, List, Optional

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()


def get_agent_client(
    project_endpoint: Optional[str] = None,
    agent_name: Optional[str] = None,
    allow_preview: bool = True,
):
    """Return an OpenAI-compatible client pointed at a deployed Foundry agent."""
    resolved_endpoint = project_endpoint or os.getenv("AZURE_AI_PROJECT_ENDPOINT")
    resolved_agent_name = agent_name or os.getenv("FOUNDRY_AGENT_NAME", "imac-agent-new")

    print(f"[DEBUG] Foundry agent setup: endpoint={'set' if resolved_endpoint else 'unset'}, agent_name={resolved_agent_name}")

    if not resolved_endpoint:
        print("[DEBUG] Missing AZURE_AI_PROJECT_ENDPOINT environment variable")
        raise RuntimeError(
            "AZURE_AI_PROJECT_ENDPOINT is required. It should point to your Azure AI Project endpoint."
        )

    print(f"[DEBUG] Creating AIProjectClient with endpoint: {resolved_endpoint}")
    project_client = AIProjectClient(
        endpoint=resolved_endpoint,
        credential=DefaultAzureCredential(),
        allow_preview=allow_preview,
    )

    print(f"[DEBUG] Requesting OpenAI-compatible client for agent: {resolved_agent_name}")
    return project_client.get_openai_client(agent_name=resolved_agent_name)


def build_messages(
    prompt: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Convert stored chat history into the message format expected by the agent responses API."""
    messages: List[Dict[str, str]] = []

    for message in conversation_history or []:
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            continue
        if not isinstance(content, str):
            continue
        stripped = content.strip()
        if stripped:
            messages.append({"role": role, "content": stripped})

    if prompt:
        stripped_prompt = prompt.strip()
        if not messages or messages[-1].get("role") != "user" or messages[-1].get("content") != stripped_prompt:
            messages.append({"role": "user", "content": stripped_prompt})

    return messages


def extract_agent_text(response) -> str:
    """Extract the final answer text from the agent response object."""
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    for item in getattr(response, "output", []) or []:
        item_type = getattr(item, "type", None)
        if item_type == "message":
            content = getattr(item, "content", None)
            if isinstance(content, list):
                texts = []
                for block in content:
                    if getattr(block, "type", None) == "output_text":
                        text = getattr(block, "text", None)
                        if isinstance(text, str) and text.strip():
                            texts.append(text.strip())
                if texts:
                    return "\n".join(texts)

    raise RuntimeError("Foundry agent returned no usable text content.")


def ask_foundry_agent(
    prompt: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    project_endpoint: Optional[str] = None,
    agent_name: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
):
    """Ask a deployed Foundry agent for an answer and return the generated text."""
    print("[DEBUG] ask_foundry_agent starting")
    agent_client = get_agent_client(
        project_endpoint=project_endpoint,
        agent_name=agent_name,
    )

    resolved_model = model or os.getenv("FOUNDRY_AGENT_MODEL", "gpt-5-mini")
    resolved_max_tokens = max_tokens or int(os.getenv("FOUNDRY_AGENT_MAX_TOKENS", "800"))
    prepared_messages = build_messages(prompt, conversation_history)

    print(f"[DEBUG] Agent request: model={resolved_model}, max_output_tokens={resolved_max_tokens}, message_count={len(prepared_messages)}")
    print(f"[DEBUG] Agent messages preview: {prepared_messages[:3]}")

    try:
        response = agent_client.responses.create(
            model=resolved_model,
            input=prepared_messages,
            max_output_tokens=resolved_max_tokens,
        )
    except Exception:
        print("[DEBUG] Foundry agent responses API call failed")
        traceback.print_exc()
        raise

    print(f"[DEBUG] Agent response object type: {type(response).__name__}")
    print(f"[DEBUG] Agent response status: {getattr(response, 'status', None)}")
    print(f"[DEBUG] Agent response output item count: {len(getattr(response, 'output', []) or [])}")

    answer = extract_agent_text(response)
    if not answer:
        print("[DEBUG] Foundry agent returned empty content")
        raise RuntimeError("Foundry agent returned an empty response.")

    print("[DEBUG] Foundry agent returned a response")
    return answer


def streamlit_agent_response(
    prompt: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    project_endpoint: Optional[str] = None,
    agent_name: Optional[str] = None,
):
    """Convenience wrapper for a Streamlit frontend.

    Returns the answer from the deployed Foundry agent so the frontend can render it.
    """
    print("[DEBUG] streamlit_agent_response invoked")
    try:
        answer = ask_foundry_agent(
            prompt=prompt,
            conversation_history=conversation_history,
            project_endpoint=project_endpoint,
            agent_name=agent_name,
        )
        print("[DEBUG] streamlit_agent_response succeeded")
        return answer
    except Exception:
        print("[DEBUG] streamlit_agent_response failed")
        traceback.print_exc()
        raise


# Example environment variables to set:
# AZURE_AI_PROJECT_ENDPOINT=https://<your-project>.services.ai.azure.com/api/projects/<project-name>
# FOUNDRY_AGENT_NAME=imac-agent-new
# FOUNDRY_AGENT_MODEL=gpt-5-mini
# FOUNDRY_AGENT_MAX_TOKENS=800