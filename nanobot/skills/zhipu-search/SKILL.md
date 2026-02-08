---
name: zhipu-search
description: Search the web using Zhipu AI (GLM) web search. Best for Chinese content, supports 1-50 results. Use for real-time info, news, weather, or any web search needs.
homepage: https://docs.bigmodel.cn/cn/guide/tools/web-search.md
always: true
metadata: {"nanobot":{"emoji":"🔍","always":true}}
---

# Zhipu Web Search

Use the `zhipu_web_search` tool for web searches with excellent Chinese content support.

## Basic Usage

Simple search:
```python
zhipu_web_search(query="search query", count=5)
```

Examples:
```python
# News search
zhipu_web_search(query="AI latest developments 2026", count=5)

# Weather query
zhipu_web_search(query="Beijing weather today", count=3)

# Technical docs
zhipu_web_search(query="Python asyncio tutorial", count=10)
```

## Search Engines

Four engine types available:

```python
zhipu_web_search(
    query="search query",
    count=10,
    search_engine="search_pro"  # recommended
)
```

**Engine options**:
- `search_std` - Standard (¥0.01/query, basic)
- `search_pro` - Professional (¥0.03/query, multi-engine, recommended)
- `search_pro_sogou` - Sogou Pro (¥0.05/query, specialized)
- `search_pro_quark` - Quark Pro (¥0.05/query, specialized)

## Result Count

Supports 1-50 results (vs 10 for other search tools):

```python
# Deep research with many results
zhipu_web_search(query="deep learning frameworks", count=30)
```

## Use Cases

**Real-time info**:
```python
zhipu_web_search(query="latest tech news", count=10)
zhipu_web_search(query="stock market today", count=5)
```

**Weather**:
```python
zhipu_web_search(query="Shanghai weather today", count=1)
zhipu_web_search(query="Beijing weather forecast", count=3)
```

**Technical docs**:
```python
zhipu_web_search(query="FastAPI documentation", count=5)
zhipu_web_search(query="React Hooks best practices", count=10)
```

**Academic research**:
```python
zhipu_web_search(query="Transformer architecture paper", count=15)
zhipu_web_search(query="quantum computing research", count=20)
```

## Output Format

Results include:
- **Title**: Page title
- **Link**: Page URL
- **Snippet**: Content excerpt (~200 chars)
- **Date**: Publication date (if available)
- **Source**: Source website

Example output:
```
Results for: Beijing weather today

1. Beijing Weather Forecast
   https://weather.example.com/beijing
   Today's weather in Beijing is sunny with rising temperatures...
   Published: 2026-02-09

2. China Weather Network - Beijing
   https://weather.china.com/beijing
   Beijing Meteorological Bureau forecast: Sunny during the day...
   Published: 2026-02-09
```

## Key Features

1. **Chinese optimized**: Excellent for Chinese content
2. **Rich results**: 1-50 results supported
3. **Multi-engine**: search_pro aggregates multiple engines
4. **Real-time**: Includes publication dates
5. **No extra config**: Uses Zhipu API key only

## Cost & Tips

- **API cost**: ¥0.01-0.05 per query (by engine type)
- **Result count**: Set reasonable count to avoid waste
- **Engine choice**:
  - General queries: `search_pro` (recommended)
  - Cost-sensitive: `search_std`
  - Specialized: `search_pro_sogou` or `search_pro_quark`

## Quick Reference

```python
# Basic (recommended)
zhipu_web_search(query="search query", count=5)

# Professional (more results)
zhipu_web_search(query="search query", count=10, search_engine="search_pro")

# Deep research (many results)
zhipu_web_search(query="search query", count=30, search_engine="search_pro")

# Economy mode (lower cost)
zhipu_web_search(query="search query", count=5, search_engine="search_std")
```

## Comparison

| Feature | zhipu_web_search | web_search (Brave) |
|---------|------------------|-------------------|
| Chinese support | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐ Fair |
| Result count | 1-50 | 1-10 |
| Publication date | ✅ Yes | ❌ No |
| Multi-engine | ✅ Yes | ❌ No |
| API key | Zhipu API Key | Brave API Key |
| Cost | ¥0.01-0.05/query | ~$5/month (2000) |

## Related Docs

- Zhipu Web Search API: https://docs.bigmodel.cn/cn/guide/tools/web-search.md
- API key: https://open.bigmodel.cn/usercenter/apikeys
- Config: `~/.nanobot/config.json`
