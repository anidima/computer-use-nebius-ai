"""
Agentic sampling loop that calls the Anthropic API and local implementation of anthropic-defined computer use tools.
"""

import platform
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Any, cast

import httpx
import json
from anthropic import (
    Anthropic,
    AnthropicBedrock,
    AnthropicVertex,
    APIError,
    APIResponseValidationError,
    APIStatusError,
)
from openai import OpenAI
from anthropic.types.beta import (
    BetaCacheControlEphemeralParam,
    BetaContentBlockParam,
    BetaImageBlockParam,
    BetaMessage,
    BetaMessageParam,
    BetaTextBlock,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
    BetaToolUseBlockParam,
)

from .tools import (
    TOOL_GROUPS_BY_VERSION,
    ToolCollection,
    ToolResult,
    ToolVersion,
)

PROMPT_CACHING_BETA_FLAG = "prompt-caching-2024-07-31"


def _convert_anthropic_messages_to_openai(messages: list[BetaMessageParam]) -> list[dict]:
    """Convert Anthropic message format to OpenAI format."""
    openai_messages = []
    
    for message in messages:
        if isinstance(message["content"], str):
            # Simple text message
            openai_messages.append({
                "role": message["role"],
                "content": message["content"]
            })
        elif isinstance(message["content"], list):
            # Complex message with multiple content blocks
            if message["role"] == "assistant":
                # Handle assistant messages with tool calls
                tool_calls = []
                content_text = ""
                
                for block in message["content"]:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            content_text += block.get("text", "")
                        elif block.get("type") == "tool_use":
                            tool_calls.append({
                                "id": block["id"],
                                "type": "function",
                                "function": {
                                    "name": block["name"],
                                    "arguments": json.dumps(block["input"])
                                }
                            })
                
                msg = {"role": "assistant"}
                if content_text:
                    msg["content"] = content_text
                if tool_calls:
                    msg["tool_calls"] = tool_calls  # type: ignore
                openai_messages.append(msg)
                
            elif message["role"] == "user":
                # Handle user messages with tool results
                content_parts = []
                tool_results = []
                
                for block in message["content"]:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            content_parts.append({"type": "text", "text": block.get("text", "")})
                        elif block.get("type") == "image":
                            content_parts.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{block['source']['media_type']};base64,{block['source']['data']}"
                                }
                            })
                        elif block.get("type") == "tool_result":
                            # Handle tool results as separate messages
                            tool_result_content = ""
                            has_image = False
                            
                            if isinstance(block.get("content"), list):
                                for content_item in block["content"]:
                                    if isinstance(content_item, dict):
                                        if content_item.get("type") == "text":
                                            tool_result_content += content_item.get("text", "")
                                        elif content_item.get("type") == "image":
                                            has_image = True
                                            # Add image as separate user message
                                            openai_messages.append({
                                                "role": "user",
                                                "content": [{
                                                    "type": "image_url",
                                                    "image_url": {
                                                        "url": f"data:{content_item['source']['media_type']};base64,{content_item['source']['data']}"
                                                    }
                                                }]
                                            })
                            elif isinstance(block.get("content"), str):
                                tool_result_content = block["content"]
                            
                            tool_results.append({
                                "role": "tool",
                                "tool_call_id": block["tool_use_id"],
                                "content": tool_result_content
                            })
                
                # Add tool results first
                openai_messages.extend(tool_results)
                
                # Add content parts if any
                if content_parts:
                    openai_messages.append({
                        "role": "user",
                        "content": content_parts if len(content_parts) > 1 else content_parts[0].get("text", "")
                    })
    
    return openai_messages


def _convert_anthropic_tools_to_openai(anthropic_tools: list) -> list[dict]:
    """Convert Anthropic tool format to OpenAI format."""
    
    # Mapping of Anthropic computer use tools to OpenAI format
    ANTHROPIC_TO_OPENAI_TOOLS = {
        "computer": {
            "type": "function",
            "function": {
                "name": "computer",
                "description": "Take a screenshot, click, type, scroll, and perform other computer actions. This tool gives you the ability to interact with the screen, keyboard, and mouse of the current computer. USE THIS TOOL FOR SCREENSHOTS with action='screenshot'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": [
                                "key", "type", "mouse_move", "left_click", "left_click_drag", 
                                "right_click", "middle_click", "double_click", "screenshot", 
                                "cursor_position", "left_mouse_down", "left_mouse_up", "scroll",
                                "hold_key", "wait", "triple_click"
                            ],
                            "description": "The action to perform"
                        },
                        "text": {
                            "type": "string",
                            "description": "Text to type (required for 'type' action)"
                        },
                        "coordinate": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": "Pixel coordinate [x, y] for mouse actions"
                        },
                        "scroll_direction": {
                            "type": "string",
                            "enum": ["up", "down", "left", "right"],
                            "description": "Direction to scroll (for 'scroll' action)"
                        }
                    },
                    "required": ["action"]
                }
            }
        },
        "str_replace_editor": {
            "type": "function",
            "function": {
                "name": "str_replace_editor",
                "description": "A text editor tool for viewing, creating and editing TEXT FILES ONLY. Can view file contents, create new files, edit files, and perform string replacements. NEVER use for screenshots or images - use computer tool for screenshots.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "enum": ["view", "create", "str_replace", "undo_edit"],
                            "description": "The command to execute"
                        },
                        "path": {
                            "type": "string",
                            "description": "Path to the file"
                        },
                        "file_text": {
                            "type": "string",
                            "description": "Text content for creating files"
                        },
                        "old_str": {
                            "type": "string",
                            "description": "String to replace (for str_replace command)"
                        },
                        "new_str": {
                            "type": "string",
                            "description": "Replacement string (for str_replace command)"
                        }
                    },
                    "required": ["command", "path"]
                }
            }
        },
        "bash": {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute bash commands in the terminal. Can run shell commands, manage files, install software, and interact with the system.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The bash command to execute"
                        }
                    },
                    "required": ["command"]
                }
            }
        }
    }
    
    openai_tools = []
    
    for tool in anthropic_tools:
        tool_name = tool.get("name")
        
        if tool_name in ANTHROPIC_TO_OPENAI_TOOLS:
            # Use predefined mapping for computer use tools
            openai_tools.append(ANTHROPIC_TO_OPENAI_TOOLS[tool_name])
        else:
            # Fallback for custom tools that might have description and input_schema
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.get("name", "unknown"),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {})
                }
            }
            openai_tools.append(openai_tool)
    
    return openai_tools


def _convert_openai_response_to_anthropic(openai_response) -> list[BetaContentBlockParam]:
    """Convert OpenAI response to Anthropic format."""
    result = []
    
    choice = openai_response.choices[0].message
    
    # Handle text content
    if choice.content:
        result.append(BetaTextBlockParam(type="text", text=choice.content))
    
    # Handle tool calls
    if choice.tool_calls:
        for tool_call in choice.tool_calls:
            result.append({
                "type": "tool_use",
                "id": tool_call.id,
                "name": tool_call.function.name,
                "input": json.loads(tool_call.function.arguments)
            })
    
    return result


class APIProvider(StrEnum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    VERTEX = "vertex"
    NEBIUS = "nebius"


# This system prompt is optimized for the Docker environment in this repository and
# specific tool combinations enabled.
# We encourage modifying this system prompt to ensure the model has context for the
# environment it is running in, and to provide any additional information that may be
# helpful for the task at hand.
SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
* You are utilising an Ubuntu virtual machine using {platform.machine()} architecture with internet access.
* You can feel free to install Ubuntu applications with your bash tool. Use curl instead of wget.
* To open firefox, please just click on the firefox icon.  Note, firefox-esr is what is installed on your system.
* ALWAYS take a screenshot FIRST to see what's currently on the desktop before trying to launch applications. Look for application icons, taskbars, or already open windows.
* TO TAKE A SCREENSHOT: Use the computer tool with action "screenshot", NOT str_replace_editor. Example: computer(action="screenshot")
* NEVER use str_replace_editor for screenshots - it is only for text files. Screenshots are done with the computer tool.
* If you see an application icon on the desktop, click on it directly instead of using bash commands to launch it.
* Using bash tool you can start GUI applications, but you need to set export DISPLAY=:1 and use a subshell. For example "(DISPLAY=:1 xterm &)". GUI apps run with bash tool will appear within your desktop environment, but they may take some time to appear. Take a screenshot to confirm it did.
* When using your bash tool with commands that are expected to output very large quantities of text, redirect into a tmp file and use str_replace_based_edit_tool or `grep -n -B <lines before> -A <lines after> <query> <filename>` to confirm output.
* When viewing a page it can be helpful to zoom out so that you can see everything on the page.  Either that, or make sure you scroll down to see everything before deciding something isn't available.
* When using your computer function calls, they take a while to run and send back to you.  Where possible/feasible, try to chain multiple of these calls all into one function calls request.
* WORKFLOW: For any task, start with a screenshot to see the current desktop state. If needed applications are visible as icons, click them. Only use bash commands as a fallback if no GUI options are available.
* IMPORTANT: When a task is completed (e.g., application opened, action performed), explicitly state "Task completed" or "Successfully opened [application]" in your response to indicate completion.
* The current date is {datetime.today().strftime('%A, %B %d, %Y')}.
</SYSTEM_CAPABILITY>

<IMPORTANT>
* CRITICAL: str_replace_editor is ONLY for text files (.txt, .py, .js, etc). NEVER use it for images or screenshots.
* For screenshots: ALWAYS use computer tool with action="screenshot". 
* When using Firefox, if a startup wizard appears, IGNORE IT.  Do not even click "skip this step".  Instead, click on the address bar where it says "Search or enter address", and enter the appropriate search term or URL there.
* If the item you are looking at is a pdf, if after taking a single screenshot of the pdf it seems that you want to read the entire document instead of trying to continue to read the pdf from your screenshots + navigation, determine the URL, use curl to download the pdf, install and use pdftotext to convert it to a text file, and then read that text file directly with your str_replace_based_edit_tool.
</IMPORTANT>"""


def _make_nebius_tool_result(result: ToolResult, tool_call_id: str, tool_name: str) -> list[dict]:
    """Convert a ToolResult to OpenAI format messages."""
    messages = []
    
    # Create tool response message
    tool_content = ""
    if result.error:
        tool_content = _maybe_prepend_system_tool_result(result, result.error)
    elif result.output:
        tool_content = _maybe_prepend_system_tool_result(result, result.output)
    
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": tool_name,
        "content": tool_content
    })
    
    # If there's an image, add it as a separate user message
    if result.base64_image:
        messages.append({
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{result.base64_image}"
                }
            }]
        })
    
    return messages


async def sampling_loop(
    *,
    model: str,
    provider: APIProvider,
    system_prompt_suffix: str,
    messages: list[BetaMessageParam],
    output_callback: Callable[[BetaContentBlockParam], None],
    tool_output_callback: Callable[[ToolResult, str], None],
    api_response_callback: Callable[
        [httpx.Request, httpx.Response | object | None, Exception | None], None
    ],
    api_key: str,
    only_n_most_recent_images: int | None = None,
    max_tokens: int = 4096,
    tool_version: ToolVersion,
    thinking_budget: int | None = None,
    token_efficient_tools_beta: bool = False,
):
    """
    Agentic sampling loop for the assistant/tool interaction of computer use.
    """
    tool_group = TOOL_GROUPS_BY_VERSION[tool_version]
    tool_collection = ToolCollection(*(ToolCls() for ToolCls in tool_group.tools))
    system = BetaTextBlockParam(
        type="text",
        text=f"{SYSTEM_PROMPT}{' ' + system_prompt_suffix if system_prompt_suffix else ''}",
    )

    while True:
        enable_prompt_caching = False
        betas = [tool_group.beta_flag] if tool_group.beta_flag else []
        if token_efficient_tools_beta:
            betas.append("token-efficient-tools-2025-02-19")
        image_truncation_threshold = only_n_most_recent_images or 0
        client = None
        if provider == APIProvider.ANTHROPIC:
            client = Anthropic(api_key=api_key, max_retries=4)
            enable_prompt_caching = True
        elif provider == APIProvider.VERTEX:
            client = AnthropicVertex()
        elif provider == APIProvider.BEDROCK:
            client = AnthropicBedrock()
        elif provider == APIProvider.NEBIUS:
            client = OpenAI(
                base_url="https://api.studio.nebius.com/v1/",
                api_key=api_key
            )

        if enable_prompt_caching:
            betas.append(PROMPT_CACHING_BETA_FLAG)
            _inject_prompt_caching(messages)
            # Because cached reads are 10% of the price, we don't think it's
            # ever sensible to break the cache by truncating images
            only_n_most_recent_images = 0
            # Use type ignore to bypass TypedDict check until SDK types are updated
            system["cache_control"] = {"type": "ephemeral"}  # type: ignore

        if only_n_most_recent_images:
            _maybe_filter_to_n_most_recent_images(
                messages,
                only_n_most_recent_images,
                min_removal_threshold=image_truncation_threshold,
            )
        extra_body = {}
        if thinking_budget:
            # Ensure we only send the required fields for thinking
            extra_body = {
                "thinking": {"type": "enabled", "budget_tokens": thinking_budget}
            }

        # Call the API
        # we use raw_response to provide debug information to streamlit. Your
        # implementation may be able call the SDK directly with:
        # `response = client.messages.create(...)` instead.
        try:
            if provider == APIProvider.NEBIUS:
                # Convert messages and tools to OpenAI format
                openai_messages = _convert_anthropic_messages_to_openai(messages)
                openai_tools = _convert_anthropic_tools_to_openai(tool_collection.to_params())
                
                # Add system message at the beginning for OpenAI
                system_content = f"{SYSTEM_PROMPT}{' ' + system_prompt_suffix if system_prompt_suffix else ''}"
                openai_messages.insert(0, {"role": "system", "content": system_content})
                
                response = client.chat.completions.create(
                    model=model,
                    messages=openai_messages,
                    tools=openai_tools if openai_tools else None,
                    tool_choice="required",
                    max_tokens=max_tokens
                )

                
                # Create mock raw_response for compatibility
                class MockHttpResponse:
                    def __init__(self):
                        self.request = None
                
                class MockRawResponse:
                    def __init__(self, response):
                        self.response = response
                        self.http_response = MockHttpResponse()
                    def parse(self):
                        return self.response
                
                raw_response = MockRawResponse(response)
            else:
                # Original Anthropic API call
                raw_response = client.beta.messages.with_raw_response.create(
                    max_tokens=max_tokens,
                    messages=messages,
                    model=model,
                    system=[system],
                    tools=tool_collection.to_params(),
                    betas=betas,
                    extra_body=extra_body,
                )
        except (APIStatusError, APIResponseValidationError) as e:
            api_response_callback(e.request, e.response, e)
            return messages
        except APIError as e:
            api_response_callback(e.request, e.body, e)
            return messages
        except Exception as e:
            # Handle OpenAI and other errors
            try:
                api_response_callback(None, None, e)  # type: ignore
            except:
                pass
            return messages

        # Handle API response callback with compatibility for both Anthropic and OpenAI
        try:
            if provider == APIProvider.NEBIUS:
                # For Nebius, we create a minimal callback
                api_response_callback(None, response, None)  # type: ignore
            else:
                # For Anthropic, use original callback
                api_response_callback(
                    raw_response.http_response.request,  # type: ignore
                    raw_response.http_response,  # type: ignore
                    None
                )
        except:
            # Ignore callback errors for compatibility
            pass

        response = raw_response.parse()

        if provider == APIProvider.NEBIUS:
            response_params = _convert_openai_response_to_anthropic(response)
        else:
            response_params = _response_to_params(response)
        messages.append(
            {
                "role": "assistant",
                "content": response_params,
            }
        )

        tool_result_content: list[BetaToolResultBlockParam] = []
        nebius_tool_messages: list[dict] = []
        
        for content_block in response_params:
            output_callback(content_block)
            if content_block["type"] == "tool_use":
                result = await tool_collection.run(
                    name=content_block["name"],
                    tool_input=cast(dict[str, Any], content_block["input"]),
                )
                
                if provider == APIProvider.NEBIUS:
                    # For Nebius, collect tool result messages
                    tool_messages = _make_nebius_tool_result(
                        result, 
                        content_block["id"], 
                        content_block["name"]
                    )
                    nebius_tool_messages.extend(tool_messages)
                else:
                    # For Anthropic, use original format
                    tool_result_content.append(
                        _make_api_tool_result(result, content_block["id"])
                    )
                
                tool_output_callback(result, content_block["id"])

        if provider == APIProvider.NEBIUS:
            if nebius_tool_messages:
                # For Nebius, extend the messages list with tool results
                messages.extend(nebius_tool_messages)
            else:
                return messages
        else:
            if not tool_result_content:
                return messages
            messages.append({"content": tool_result_content, "role": "user"})


def _maybe_filter_to_n_most_recent_images(
    messages: list[BetaMessageParam],
    images_to_keep: int,
    min_removal_threshold: int,
):
    """
    With the assumption that images are screenshots that are of diminishing value as
    the conversation progresses, remove all but the final `images_to_keep` tool_result
    images in place, with a chunk of min_removal_threshold to reduce the amount we
    break the implicit prompt cache.
    """
    if images_to_keep is None:
        return messages

    tool_result_blocks = cast(
        list[BetaToolResultBlockParam],
        [
            item
            for message in messages
            for item in (
                message["content"] if isinstance(message["content"], list) else []
            )
            if isinstance(item, dict) and item.get("type") == "tool_result"
        ],
    )

    total_images = sum(
        1
        for tool_result in tool_result_blocks
        for content in tool_result.get("content", [])
        if isinstance(content, dict) and content.get("type") == "image"
    )

    images_to_remove = total_images - images_to_keep
    # for better cache behavior, we want to remove in chunks
    images_to_remove -= images_to_remove % min_removal_threshold

    for tool_result in tool_result_blocks:
        if isinstance(tool_result.get("content"), list):
            new_content = []
            for content in tool_result.get("content", []):
                if isinstance(content, dict) and content.get("type") == "image":
                    if images_to_remove > 0:
                        images_to_remove -= 1
                        continue
                new_content.append(content)
            tool_result["content"] = new_content


def _response_to_params(
    response: BetaMessage,
) -> list[BetaContentBlockParam]:
    res: list[BetaContentBlockParam] = []
    for block in response.content:
        if isinstance(block, BetaTextBlock):
            if block.text:
                res.append(BetaTextBlockParam(type="text", text=block.text))
            elif getattr(block, "type", None) == "thinking":
                # Handle thinking blocks - include signature field
                thinking_block = {
                    "type": "thinking",
                    "thinking": getattr(block, "thinking", None),
                }
                if hasattr(block, "signature"):
                    thinking_block["signature"] = getattr(block, "signature", None)
                res.append(cast(BetaContentBlockParam, thinking_block))
        else:
            # Handle tool use blocks normally
            res.append(cast(BetaToolUseBlockParam, block.model_dump()))
    return res


def _inject_prompt_caching(
    messages: list[BetaMessageParam],
):
    """
    Set cache breakpoints for the 3 most recent turns
    one cache breakpoint is left for tools/system prompt, to be shared across sessions
    """

    breakpoints_remaining = 3
    for message in reversed(messages):
        if message["role"] == "user" and isinstance(
            content := message["content"], list
        ):
            if breakpoints_remaining:
                breakpoints_remaining -= 1
                # Use type ignore to bypass TypedDict check until SDK types are updated
                content[-1]["cache_control"] = BetaCacheControlEphemeralParam(  # type: ignore
                    {"type": "ephemeral"}
                )
            else:
                content[-1].pop("cache_control", None)
                # we'll only every have one extra turn per loop
                break


def _make_api_tool_result(
    result: ToolResult, tool_use_id: str
) -> BetaToolResultBlockParam:
    """Convert an agent ToolResult to an API ToolResultBlockParam."""
    tool_result_content: list[BetaTextBlockParam | BetaImageBlockParam] | str = []
    is_error = False
    if result.error:
        is_error = True
        tool_result_content = _maybe_prepend_system_tool_result(result, result.error)
    else:
        if result.output:
            tool_result_content.append(
                {
                    "type": "text",
                    "text": _maybe_prepend_system_tool_result(result, result.output),
                }
            )
        if result.base64_image:
            tool_result_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": result.base64_image,
                    },
                }
            )
    return {
        "type": "tool_result",
        "content": tool_result_content,
        "tool_use_id": tool_use_id,
        "is_error": is_error,
    }


def _maybe_prepend_system_tool_result(result: ToolResult, result_text: str):
    if result.system:
        result_text = f"<system>{result.system}</system>\n{result_text}"
    return result_text
