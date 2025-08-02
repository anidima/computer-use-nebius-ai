# Nebius AI Computer Use Demo

This repository contains an adapted version of Anthropic's Computer Use Demo that has been modified to work with Nebius AI's vision-capable models. The core functionality of the computer use agent has been preserved while replacing the Anthropic API calls with equivalent Nebius AI API calls.

## Overview

The Nebius AI adaptation maintains all the original computer use capabilities:
- Screenshot capture and visual analysis
- Mouse control (clicking, dragging, scrolling)
- Keyboard input simulation
- File system operations
- Terminal/bash command execution
- Browser automation

The main difference is that the agent now uses Nebius AI's `Qwen/Qwen2.5-VL-72B-Instruct` model for vision instead of Claude models.

## Setup instructions

### Prerequisites
- Docker installed on your system
- Nebius AI API key (obtain from [Nebius Studio](https://studio.nebius.com/))

### Environment setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd anthropic-quickstarts/computer-use-demo
   ```

2. **Set environment variables**:
   ```bash
   export NEBIUS_API_KEY="your_api_key"
   ```

3. **Build and run the Docker container**:
   ```bash
   docker build -t nebius-computer-use .
   
   docker run \
       -e NEBIUS_API_KEY=$NEBIUS_API_KEY \
       -e API_PROVIDER=nebius \
       -v $HOME/.anthropic:/home/computeruse/.anthropic \
       -p 5900:5900 \
       -p 8501:8501 \
       -p 6080:6080 \
       -p 8080:8080 \
       -it nebius-computer-use
   ```

### Open streamlit interface

http://localhost:8080



## Adapting Anthropic’s computer-use-demo to the Nebius AI


First I studied the structure of the **computer-use-demo** repository:

- `streamlit.py` – the web interface that surfaces configuration options and displays model output.
- `loop.py` – the core execution engine that defines the system prompt, the `APIProvider` class, and the asynchronous `sampling_loop` responsible for exchanging messages with the language model and dispatching tool calls.

---

### 1 | Integrating Nebius AI

The first thing was to add the `APIProvider.NEBIUS` and create a client for it:

```python
elif provider == APIProvider.NEBIUS:
    client = OpenAI(
        base_url="https://api.studio.nebius.com/v1/",
        api_key=api_key,
    )
```

Nebius supplies an OpenAI-compatible SDK, so only the base URL and key management required modification.

---

### 2 | Reconciling message formats

Anthropic relies on structured *Beta* blocks (`BetaTextBlockParam`, `BetaImageBlockParam`, `BetaToolUseBlockParam`, etc.), whereas Nebius follows the OpenAI Chat Completion schema: a sequential list of messages with an optional `tool_calls` array.

Obviously, it is necessary to transform queries before feeding them to the model and model responses in order to establish compatibility.
Ideally, it would be necessary to create an abstraction layer, create adapter classes separately for each API and change sampling_loop to work only with these classes. But for nebius models, a different response request formation, different error handling, etc. will also be needed.

To avoid a broad refactor, I implemented a conversion layer with three functions that operates at the network boundary:

- *_convert_anthropic_messages_to_openai* – flattens Anthropic blocks into OpenAI-style message objects and ensures the system prompt is the first entry.
- *_convert_anthropic_tools_to_openai* – maps the existing *computer*, *bash*, and *str\_replace\_editor* tool definitions to OpenAI function-calling JSON.
- *_convert_openai_response_to_anthropic* – restores Nebius responses to Anthropic block format so that downstream logic remains unchanged.

All other files continue to work with the original data structures.

---

### 3 | The main challenges 

#### Selecting a compatible vision model 

In the Nebius AI Studio account I had access to 4 vision models –
- google/gemma-3-27b-it
- Qwen/Qwen2-VL-72B-Instruct
- Qwen/Qwen2.5-VL-72B-Instruct
- mistralai/Mistral-Small-3.1-24B-Instruct-2503

When trying to access all these models via the suggested code in the API documentation, an error occurred stating that it is impossible to use the auto tool in these models.

```python
Error code: 400 - {'detail': 'Invalid request. Please check the parameters and try again. Details: This model does not support auto tool, please use tool_choice.'}
```
But at the same time, the documentation https://docs.nebius.com/studio/inference/tool-calling states that
The tool_choice parameter defines the function selection logic. Supported values:
auto: the most suitable function is selected based on the context of the prompt.
Specific function name: {"type": "function", "function": {"name": "read_file"}}.

But at the same time, the documentation https://docs.nebius.com/studio/inference/tool-calling states that
```python
The tool_choice parameter defines the function selection logic. Supported values:
auto: the most suitable function is selected based on the context of the prompt.
Specific function name: {"type": "function", "function": {"name": "read_file"}}.
```

I created tests for all 4 models and tested variants 
`without tool_choice`, `tool_choice=None`, `tool_choice=“none”`, `tool_choice=“auto”` and `tool_choice=“required”`.

As a result, I found out that Qwen/Qwen2.5-VL-72B-Instruct accepts `tool_choice="required"`, `tool_choice="none"`
Qwen/Qwen2-VL-72B-Instruct accepts `tool_choice="none"`
And the other two return an error.

I left the Qwen/Qwen2.5-VL-72B-Instruct model with the `tool_choice="required"` parameter, but since this value requires the model to select a tool in the response, accessing the model turns into an infinite loop. It is worth noting that the response to the first request is returned relevant and processed correctly. I tested the screenshot and running the command via bash.

---

#### Refining the system prompt

Several prompt adjustments were necessary to align the model’s behaviour with the original demo:

1. Always capture an initial screenshot before reasoning.
2. Prefer the most direct tool (for example, a mouse click) before resorting to Bash.

These changes reduced unnecessary tool usage and kept responses concise.

#### Container reliability

To build the Docker image on Windows, I modified the `Dockerfile` to now install `dos2unix` and explicitly set execute permissions for all shell scripts. This prevents runtime errors related to line termination.
 

---

## Outcome

With the conversion layer and prompt adjustments in place, the agent now operates with Nebius AI while retaining the original workflow: it captures a screenshot, reasons about the interface, selects or omits tools as needed, and iterates until the task is completed. 

---

## Further possible steps

- Identify a vision model that can accept `tool_choice="auto"` in the request.
- Let a separate model handle tool orchestration, while the vision model focuses on screenshot recognition - connect the two so they work together seamlessly.
- Create a dedicated provider-adapter abstraction to make it easier to plug in additional APIs later on.

---

## Evaluation 

From the user's point of view, first of all, I would measure whether the agent manages to achieve the requested result - `task completion`. For example, the number of successfully completed tasks to the total number.

Then the following criteria should be assessed upon successful task completion.
One of them is the `speed of achieving the result` - there are several factors here: how accurately planning occurs, how quickly the model copes with thinking and processing information, how often repeated calls to the same tools occur, hallucinations of models, the length of the agent's trajectory.

`The cost` of successfully completing a task - it is more profitable to use a smaller model if the accuracy and speed of completing the task are proportionate.

`The accuracy of the tool selection` is also important if the user explicitly indicated this in the request. For example, to solve the example 2 * 2: the LLM model itself can calculate, or it can call the echo $ ((2 * 2)) command, or it can open the calculator and calculate there.

Each of these high-level metrics can be broken down into smaller, more specific metrics so that you can focus on one or the other.