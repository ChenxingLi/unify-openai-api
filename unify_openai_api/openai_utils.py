def split_openai_params(data) -> tuple[dict, dict]:
    supported_fields = {
        "messages",
        "model",
        "audio",
        "frequency_penalty",
        "function_call",
        "functions",
        "logit_bias",
        "logprobs",
        "max_completion_tokens",
        "max_tokens",
        "metadata",
        "modalities",
        "n",
        "parallel_tool_calls",
        "prediction",
        "presence_penalty",
        "reasoning_effort",
        "response_format",
        "seed",
        "service_tier",
        "stop",
        "store",
        "stream",
        "stream_options",
        "temperature",
        "tool_choice",
        "tools",
        "top_logprobs",
        "top_p",
        "user",
        "extra_headers",
        "extra_query",
        "extra_body",
        "timeout"
    }
    supported, unsupported = {}, {}
    for key, value in data.items():
        if key in supported_fields:
            supported[key] = value
        else:
            unsupported[key] = value
    return supported, unsupported