import os

PRIMARY_MODEL_ID = os.getenv("LOCAL_PRIMARY_MODEL_NAME", "gpt-4.1")
RESEARCH_MODEL_ID = os.getenv("LOCAL_RESEARCH_MODEL_NAME", PRIMARY_MODEL_ID)
BROWSER_MODEL_ID = os.getenv("LOCAL_BROWSER_MODEL_NAME", PRIMARY_MODEL_ID)
ANALYZER_MODEL_ID = os.getenv(
    "LOCAL_ANALYZER_MODEL_NAME",
    os.getenv("LOCAL_VLLM_MODEL_NAME", "local-vllm"),
)
SUMMARIZER_MODEL_ID = os.getenv("LOCAL_SUMMARIZER_MODEL_NAME", PRIMARY_MODEL_ID)
DEEP_RESEARCH_TOOL_MODEL_ID = os.getenv(
    "LOCAL_DEEP_RESEARCH_MODEL_NAME",
    "o3-deep-research",
)

web_fetcher_tool_config = dict(
    type="web_fetcher_tool",
)

web_searcher_tool_config = dict(
    type="web_searcher_tool",
    engine="Firecrawl",  # Options: "Firecrawl", "Google", "Bing", "DuckDuckGo", "Baidu"
    retry_delay = 10,
    max_retries = 3,
    lang = "en",
    country = "us",
    num_results = 5,
    fetch_content = True,
    max_length = 4096,
)

deep_researcher_tool_config  = dict(
    type="deep_researcher_tool",
    model_id = RESEARCH_MODEL_ID,
    max_depth = 2,
    max_insights = 20,
    time_limit_seconds = 60,
    max_follow_ups = 3,
)

auto_browser_use_tool_config  = dict(
    type="auto_browser_use_tool",
    model_id=BROWSER_MODEL_ID,
)

deep_analyzer_tool_config  = dict(
    type="deep_analyzer_tool",
    analyzer_model_ids = [ANALYZER_MODEL_ID],
    summarizer_model_id = SUMMARIZER_MODEL_ID,
)

mcp_tools_config = {
    "mcpServers" :  {
        # Local stdio server
        "LocalMCP": {
            "command": "python",
            "args": ["src/mcp/server.py"],
            "env": {"DEBUG": "true"}
        },
        # Remote server
        # "calendar": {
        #     "url": "https://calendar-api.example.com/mcp",
        #     "transport": "streamable-http"
        # }
    }
}

image_generator_tool_config = dict(
    type="image_generator_tool",
    analyzer_model_id = "o3",
    generator_model_id = "imagen",
)

video_generator_tool_config = dict(
    type="video_generator_tool",
    analyzer_model_id = "o3",
    predict_model_id = "veo3-predict",
    fetch_model_id = "veo3-fetch",
)

file_reader_tool_config = dict(
    type="file_reader_tool"
)

oai_deep_research_tool_config = dict(
    type="oai_deep_research_tool",
    model_id = DEEP_RESEARCH_TOOL_MODEL_ID,
)
