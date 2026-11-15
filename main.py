#!/usr/bin/env python3
"""free-model-aggregator-01: one server, two client wires, ten free-tier providers.

Serves the OpenAI Chat Completions API and the Anthropic Messages API (JSON and
server-sent events) over the providers named in the trial brief, routing every
request to the best free model that meets its requirements and failing over to
the next candidate when a provider refuses (402/429/5xx). Registers its port
and catalog with the env at $MAKEFUN_ENV_URL and exits 0 when the env's client
is done. `python3 main.py --list` prints the catalog document, byte for byte
the document the register body carries.

Standard library only. Every choice is a pure function of the brief, the
requests seen so far and the providers' answers: no wall clock, no randomness,
so a replayed trial leaves the same state export.
"""

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection, HTTPSConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

PROTOCOL_VERSION = 1
POLL_INTERVAL_S = 0.1
PROBE_TIMEOUT_S = 5.0
UPSTREAM_TIMEOUT_S = 20.0
MAX_BODY_BYTES = 32 * 1024 * 1024

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

FIXED_OPENAI_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"
ANTHROPIC_VERSION = "2023-06-01"

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def is_record(v):
    return isinstance(v, dict)


def string_of(v):
    return v if isinstance(v, str) else ""


def number_or_none(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and v not in (float("inf"), float("-inf")) else None


def bool_or_none(v):
    return v if isinstance(v, bool) else None


# --------------------------------------------------------------------------
# The brief: three JSON documents found by their `schema` member.
# --------------------------------------------------------------------------

def _object_end(text, start):
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def documents_in(text):
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except ValueError:
                value = None
            if is_record(value) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def provider_of(v):
    if not is_record(v):
        return None
    pid = string_of(v.get("id"))
    wire = v.get("wire")
    if pid == "" or wire not in ("openai", "anthropic", "gemini"):
        return None
    auth = v.get("auth")
    if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
        auth = "bearer"
    base = ""
    for key in ("path", "base_path", "base_url", "base"):
        if string_of(v.get(key)) != "":
            base = string_of(v.get(key))
            break
    return {
        "id": pid,
        "name": string_of(v.get("name")) or pid,
        "wire": wire,
        "auth": auth,
        "base": base,
        "key_env": string_of(v.get("key_env")),
        "key": string_of(v.get("key")),
    }


def rating_of(v):
    if not is_record(v):
        return None
    model = string_of(v.get("model"))
    provider = string_of(v.get("provider"))
    if model == "" or provider == "":
        return None
    rating = v.get("rating_milli")
    return {
        "model": model,
        "provider": provider,
        "rating_milli": rating if isinstance(rating, (int, float)) and not isinstance(rating, bool) else 0,
        "context_length": number_or_none(v.get("context_length")),
        "tools": v.get("tools") is True,
        "vision": v.get("vision") is True,
        "json": v.get("json") is True,
        "free": v.get("free") is True,
    }


def workload_request_of(v):
    if not is_record(v):
        return None
    rid = string_of(v.get("id"))
    if rid == "":
        return None
    requires = v.get("requires") if is_record(v.get("requires")) else {}
    req = {
        "tools": requires.get("tools") is True,
        "vision": requires.get("vision") is True,
        "json": requires.get("json") is True,
    }
    if number_or_none(requires.get("min_context")) is not None:
        req["min_context"] = requires["min_context"]
    return {
        "id": rid,
        "api": "anthropic" if v.get("api") == "anthropic" else "openai",
        "stream": v.get("stream") is True,
        "weight": 2 if v.get("weight") == 2 else 1,
        "requires": req,
    }


class Brief:
    def __init__(self):
        self.providers = []
        self.ratings = None
        self.workload = None
        self.error = None


def read_brief(path):
    brief = Brief()
    if path == "":
        brief.error = "MAKEFUN_PROMPT_FILE is not set"
        return brief
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as err:
        brief.error = "cannot read %s: %s" % (path, err)
        return brief
    docs = documents_in(text)
    directory_doc = None
    ratings_doc = None
    workload_doc = None
    for doc in docs:
        if doc.get("schema") == PROVIDERS_SCHEMA and isinstance(doc.get("providers"), list) and directory_doc is None:
            directory_doc = doc
        elif doc.get("schema") == RATINGS_SCHEMA and isinstance(doc.get("ratings"), list) and ratings_doc is None:
            ratings_doc = doc
        elif doc.get("schema") == WORKLOAD_SCHEMA and isinstance(doc.get("manifest"), list) and workload_doc is None:
            workload_doc = doc
    if directory_doc is None:
        brief.error = "the brief at %s holds no %s document" % (path, PROVIDERS_SCHEMA)
        return brief
    brief.providers = [p for p in (provider_of(v) for v in directory_doc["providers"]) if p is not None]
    if ratings_doc is not None:
        brief.ratings = [r for r in (rating_of(v) for v in ratings_doc["ratings"]) if r is not None]
    if workload_doc is not None:
        brief.workload = [w for w in (workload_request_of(v) for v in workload_doc["manifest"]) if w is not None]
    return brief


def rating_row(brief, provider_id, model_id):
    if brief.ratings is None:
        return None
    for row in brief.ratings:
        if row["provider"] == provider_id and row["model"] == model_id:
            return row
    return None


# --------------------------------------------------------------------------
# HTTP plumbing (upstream calls and catalog probes).
# --------------------------------------------------------------------------

def join_base(env_url, base):
    b = (base or "").strip()
    if b.lower().startswith("http://") or b.lower().startswith("https://"):
        return b.rstrip("/")
    prefix = env_url.rstrip("/")
    if b == "":
        return prefix
    return prefix + ("" if b.startswith("/") else "/") + b.rstrip("/")


def auth_headers(auth, key):
    if key == "":
        return {}
    if auth == "bearer":
        return {"Authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def http_request(method, url, headers, body, timeout):
    """One HTTP call; returns (status, headers dict, body bytes)."""
    parsed = urlparse(url)
    if parsed.scheme == "https":
        conn = HTTPSConnection(parsed.hostname, parsed.port or 443, timeout=timeout)
    else:
        conn = HTTPConnection(parsed.hostname, parsed.port or 80, timeout=timeout)
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    try:
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        data = response.read()
        return response.status, {k.lower(): v for k, v in response.getheaders()}, data
    finally:
        conn.close()


# --------------------------------------------------------------------------
# The catalog probe: each provider's listing, read in its own wire.
# --------------------------------------------------------------------------

def model_in(wire, v):
    if not is_record(v):
        return None
    if wire == "gemini":
        mid = string_of(v.get("name"))
        if mid.startswith("models/"):
            mid = mid[len("models/"):]
    else:
        mid = string_of(v.get("id"))
    if mid == "":
        return None
    context = number_or_none(v.get("context_length"))
    if context is None:
        context = number_or_none(v.get("context_window"))
    if context is None:
        context = number_or_none(v.get("inputTokenLimit"))
    display = string_of(v.get("display_name")) or string_of(v.get("displayName")) or mid
    return {
        "id": mid,
        "display_name": display,
        "free": bool_or_none(v.get("free")),
        "context_length": context,
        "tools": bool_or_none(v.get("tools")),
        "vision": bool_or_none(v.get("vision")),
        "json": bool_or_none(v.get("json")),
    }


def models_in(wire, body):
    if not is_record(body):
        return []
    listing = body.get("models") if wire == "gemini" else body.get("data")
    if not isinstance(listing, list):
        return []
    return [m for m in (model_in(wire, v) for v in listing) if m is not None]


def probe_provider(env_url, provider):
    base = join_base(env_url, provider["base"])
    url = base + ("/v1beta/models" if provider["wire"] == "gemini" else "/v1/models")
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
    probe = {"provider": provider, "ok": False, "status": 0, "models": [], "error": None}
    try:
        status, _headers, data = http_request("GET", url, headers, None, PROBE_TIMEOUT_S)
        probe["status"] = status
        if status < 200 or status >= 300:
            probe["error"] = "GET %s answered HTTP %d" % (url, status)
            return probe
        body = json.loads(data.decode("utf-8"))
        probe["models"] = models_in(provider["wire"], body)
        probe["ok"] = True
        return probe
    except Exception as err:  # a failed probe leaves the provider out; the trial goes on
        probe["error"] = "GET %s failed: %s" % (url, err)
        return probe


def model_free(model, row):
    """Free per the quality table where it has a row, else the listing; paid when neither says."""
    if row is not None:
        return row["free"]
    return model["free"] if model["free"] is not None else False


def capabilities_of(model, row):
    if row is not None:
        return {
            "context_length": row["context_length"] if row["context_length"] is not None else model["context_length"],
            "tools": row["tools"],
            "vision": row["vision"],
            "json": row["json"],
        }
    return {
        "context_length": model["context_length"],
        "tools": model["tools"],
        "vision": model["vision"],
        "json": model["json"],
    }


def meets_requirements(requires, caps):
    if requires.get("tools") is True and caps["tools"] is not True:
        return False
    if requires.get("vision") is True and caps["vision"] is not True:
        return False
    if requires.get("json") is True and caps["json"] is not True:
        return False
    if requires.get("min_context") is not None:
        if caps["context_length"] is None or caps["context_length"] < requires["min_context"]:
            return False
    return True


def build_catalog(probes, brief):
    providers = []
    for probe in probes:
        if not probe["ok"]:
            continue
        seen = set()
        models = []
        for model in sorted(probe["models"], key=lambda m: m["id"]):
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            row = rating_row(brief, probe["provider"]["id"], model["id"])
            context = row["context_length"] if row is not None and row["context_length"] is not None else model["context_length"]
            models.append({"id": model["id"], "free": model_free(model, row), "context_length": context})
        providers.append({
            "id": probe["provider"]["id"],
            "base_url": probe["provider"]["base"],
            "key_env": probe["provider"]["key_env"],
            "wire": probe["provider"]["wire"],
            "models": models,
        })
    return {"providers": providers}


# --------------------------------------------------------------------------
# Wire translations. The providers echo what they parse, and the client
# compares that echo with what it sent, so every translation preserves the
# system prompt, the last user message's text and images, the tool names in
# order, max_tokens, the stop sequences and the JSON-mode flag exactly.
# --------------------------------------------------------------------------

def blocks_text(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def openai_system_text(body):
    parts = []
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    for message in messages:
        if is_record(message) and message.get("role") == "system":
            text = blocks_text(message.get("content"))
            if text != "":
                parts.append(text)
    return "\n".join(parts)


def openai_max_tokens(body):
    value = number_or_none(body.get("max_tokens"))
    if value is None:
        value = number_or_none(body.get("max_completion_tokens"))
    return value


def string_list(v):
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        return [x for x in v if isinstance(x, str)]
    return []


def parse_data_url(url):
    """A data: URL as (media_type, base64 data), or None."""
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    head, sep, data = url.partition(",")
    if sep == "" or ";base64" not in head:
        return None
    media = head[len("data:"):].split(";")[0] or "image/png"
    return media, data


def arguments_of(value):
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != "":
        try:
            parsed = json.loads(value)
            if is_record(parsed):
                return parsed
        except ValueError:
            pass
    return {}


def tool_result_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = blocks_text(content)
        return text if text != "" else dumps(content)
    if content is None:
        return ""
    return dumps(content)


# --- request builders: (client wire, client body) -> provider wire body ---

def to_openai_body(client_wire, body, model):
    if client_wire == "openai":
        out = json.loads(dumps(body))
        out["model"] = model
        if out.get("stream") is True:
            out["stream_options"] = {"include_usage": True}
        return out
    # Anthropic Messages -> OpenAI Chat Completions.
    messages = []
    system = blocks_text(body.get("system"))
    if system != "":
        messages.append({"role": "system", "content": system})
    source = body.get("messages") if isinstance(body.get("messages"), list) else []
    for raw in source:
        if not is_record(raw):
            continue
        assistant = raw.get("role") == "assistant"
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": "assistant" if assistant else "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            tool_calls = []
            for block in content:
                if not is_record(block) or block.get("type") != "tool_use":
                    continue
                tool_calls.append({
                    "id": string_of(block.get("id")),
                    "type": "function",
                    "function": {
                        "name": string_of(block.get("name")),
                        "arguments": dumps(block["input"]) if is_record(block.get("input")) else "{}",
                    },
                })
            text = blocks_text(content)
            message = {"role": "assistant", "content": text if text != "" else None}
            if tool_calls:
                message["tool_calls"] = tool_calls
            messages.append(message)
            continue
        parts = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "tool_result":
                messages.append({
                    "role": "tool",
                    "tool_call_id": string_of(block.get("tool_use_id")),
                    "content": tool_result_text(block.get("content")),
                })
                continue
            part = anthropic_part_to_openai(block)
            if part is not None:
                parts.append(part)
        if parts:
            messages.append({"role": "user", "content": parts})
    out = {"model": model, "messages": messages}
    if number_or_none(body.get("max_tokens")) is not None:
        out["max_tokens"] = body["max_tokens"]
    if number_or_none(body.get("temperature")) is not None:
        out["temperature"] = body["temperature"]
    if number_or_none(body.get("top_p")) is not None:
        out["top_p"] = body["top_p"]
    stops = string_list(body.get("stop_sequences"))
    if stops:
        out["stop"] = stops
    tools = anthropic_tools_to_openai(body.get("tools"))
    if tools:
        out["tools"] = tools
    choice = openai_tool_choice(body.get("tool_choice"))
    if choice is not None:
        out["tool_choice"] = choice
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def anthropic_part_to_openai(block):
    if block.get("type") == "text" and isinstance(block.get("text"), str):
        return {"type": "text", "text": block["text"]}
    if block.get("type") == "image" and is_record(block.get("source")):
        source = block["source"]
        if source.get("type") == "base64" and isinstance(source.get("data"), str):
            media = string_of(source.get("media_type")) or "image/png"
            return {"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (media, source["data"])}}
        if source.get("type") == "url" and isinstance(source.get("url"), str):
            return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def anthropic_tools_to_openai(tools):
    out = []
    if not isinstance(tools, list):
        return out
    for raw in tools:
        if not is_record(raw) or not isinstance(raw.get("name"), str):
            continue
        out.append({
            "type": "function",
            "function": {
                "name": raw["name"],
                "description": string_of(raw.get("description")),
                "parameters": raw["input_schema"] if is_record(raw.get("input_schema")) else {"type": "object", "properties": {}},
            },
        })
    return out


def openai_tool_choice(choice):
    if not is_record(choice):
        return None
    if choice.get("type") == "auto":
        return "auto"
    if choice.get("type") == "any":
        return "required"
    if choice.get("type") == "none":
        return "none"
    if choice.get("type") == "tool" and isinstance(choice.get("name"), str):
        return {"type": "function", "function": {"name": choice["name"]}}
    return None


def to_anthropic_body(client_wire, body, model):
    if client_wire == "anthropic":
        out = json.loads(dumps(body))
        out["model"] = model
        return out
    # OpenAI Chat Completions -> Anthropic Messages.
    out = {"model": model, "messages": []}
    system = openai_system_text(body)
    if system != "":
        out["system"] = system
    source = body.get("messages") if isinstance(body.get("messages"), list) else []
    for raw in source:
        if not is_record(raw):
            continue
        role = raw.get("role")
        if role == "system":
            continue
        content = raw.get("content")
        if role == "assistant":
            blocks = []
            text = blocks_text(content) if isinstance(content, list) else (content if isinstance(content, str) else "")
            if text != "":
                blocks.append({"type": "text", "text": text})
            calls = raw.get("tool_calls") if isinstance(raw.get("tool_calls"), list) else []
            for call in calls:
                if not is_record(call) or not is_record(call.get("function")):
                    continue
                blocks.append({
                    "type": "tool_use",
                    "id": string_of(call.get("id")),
                    "name": string_of(call["function"].get("name")),
                    "input": arguments_of(call["function"].get("arguments")),
                })
            out["messages"].append({"role": "assistant", "content": blocks if blocks else ""})
            continue
        if role == "tool":
            out["messages"].append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": string_of(raw.get("tool_call_id")),
                    "content": content if isinstance(content, str) else tool_result_text(content),
                }],
            })
            continue
        # user (or any other role): text and image parts, in order.
        if isinstance(content, str):
            out["messages"].append({"role": "user", "content": content})
            continue
        blocks = []
        if isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    blocks.append({"type": "text", "text": block["text"]})
                elif block.get("type") == "image_url" and is_record(block.get("image_url")):
                    url = block["image_url"].get("url")
                    data_url = parse_data_url(url)
                    if data_url is not None:
                        blocks.append({"type": "image", "source": {"type": "base64", "media_type": data_url[0], "data": data_url[1]}})
                    elif isinstance(url, str):
                        blocks.append({"type": "image", "source": {"type": "url", "url": url}})
        out["messages"].append({"role": "user", "content": blocks})
    max_tokens = openai_max_tokens(body)
    if max_tokens is not None:
        out["max_tokens"] = max_tokens
    if number_or_none(body.get("temperature")) is not None:
        out["temperature"] = body["temperature"]
    if number_or_none(body.get("top_p")) is not None:
        out["top_p"] = body["top_p"]
    stops = string_list(body.get("stop"))
    if stops:
        out["stop_sequences"] = stops
    tools = openai_tools_to_anthropic(body.get("tools"))
    if tools:
        out["tools"] = tools
    choice = anthropic_tool_choice(body.get("tool_choice"))
    if choice is not None:
        out["tool_choice"] = choice
    if body.get("stream") is True:
        out["stream"] = True
    return out


def openai_tools_to_anthropic(tools):
    out = []
    if not isinstance(tools, list):
        return out
    for raw in tools:
        if not is_record(raw) or not is_record(raw.get("function")) or not isinstance(raw["function"].get("name"), str):
            continue
        fn = raw["function"]
        out.append({
            "name": fn["name"],
            "description": string_of(fn.get("description")),
            "input_schema": fn["parameters"] if is_record(fn.get("parameters")) else {"type": "object", "properties": {}},
        })
    return out


def anthropic_tool_choice(choice):
    if isinstance(choice, str):
        if choice == "auto":
            return {"type": "auto"}
        if choice == "required":
            return {"type": "any"}
        if choice == "none":
            return {"type": "none"}
        return None
    if is_record(choice) and is_record(choice.get("function")) and isinstance(choice["function"].get("name"), str):
        return {"type": "tool", "name": choice["function"]["name"]}
    return None


def to_gemini_body(client_wire, body):
    """A Gemini generateContent body (the model travels in the URL path)."""
    contents = []
    declarations = []
    config = {}
    if client_wire == "openai":
        system = openai_system_text(body)
        if system != "":
            contents.append(("system", system))
        source = body.get("messages") if isinstance(body.get("messages"), list) else []
        for raw in source:
            if not is_record(raw):
                continue
            role = raw.get("role")
            if role == "system":
                continue
            content = raw.get("content")
            if role == "assistant":
                parts = []
                text = blocks_text(content) if isinstance(content, list) else (content if isinstance(content, str) else "")
                if text != "":
                    parts.append({"text": text})
                calls = raw.get("tool_calls") if isinstance(raw.get("tool_calls"), list) else []
                for call in calls:
                    if is_record(call) and is_record(call.get("function")):
                        parts.append({"functionCall": {"name": string_of(call["function"].get("name")), "args": arguments_of(call["function"].get("arguments"))}})
                contents.append(("model", parts))
                continue
            if role == "tool":
                contents.append(("user", [{"functionResponse": {"name": string_of(raw.get("tool_call_id")), "response": {"result": content if isinstance(content, str) else tool_result_text(content)}}}] ))
                continue
            parts = []
            if isinstance(content, str):
                if content != "":
                    parts.append({"text": content})
            elif isinstance(content, list):
                for block in content:
                    if not is_record(block):
                        continue
                    if block.get("type") == "text" and isinstance(block.get("text"), str):
                        parts.append({"text": block["text"]})
                    elif block.get("type") == "image_url" and is_record(block.get("image_url")):
                        url = block["image_url"].get("url")
                        data_url = parse_data_url(url)
                        if data_url is not None:
                            parts.append({"inlineData": {"mimeType": data_url[0], "data": data_url[1]}})
                        elif isinstance(url, str):
                            parts.append({"fileData": {"fileUri": url}})
            contents.append(("user", parts))
        for tool in openai_tools_to_anthropic(body.get("tools")):
            declarations.append({"name": tool["name"], "description": tool["description"], "parameters": tool["input_schema"]})
        max_tokens = openai_max_tokens(body)
        if max_tokens is not None:
            config["maxOutputTokens"] = max_tokens
        stops = string_list(body.get("stop"))
        if stops:
            config["stopSequences"] = stops
        fmt = body.get("response_format")
        if is_record(fmt) and fmt.get("type") in ("json_object", "json_schema"):
            config["responseMimeType"] = "application/json"
    else:
        system = blocks_text(body.get("system"))
        if system != "":
            contents.append(("system", system))
        source = body.get("messages") if isinstance(body.get("messages"), list) else []
        for raw in source:
            if not is_record(raw):
                continue
            role = "model" if raw.get("role") == "assistant" else "user"
            content = raw.get("content")
            parts = []
            if isinstance(content, str):
                if content != "":
                    parts.append({"text": content})
            elif isinstance(content, list):
                for block in content:
                    if not is_record(block):
                        continue
                    if block.get("type") == "text" and isinstance(block.get("text"), str):
                        parts.append({"text": block["text"]})
                    elif block.get("type") == "image" and is_record(block.get("source")):
                        source = block["source"]
                        if source.get("type") == "base64" and isinstance(source.get("data"), str):
                            parts.append({"inlineData": {"mimeType": string_of(source.get("media_type")) or "image/png", "data": source["data"]}})
                        elif source.get("type") == "url" and isinstance(source.get("url"), str):
                            parts.append({"fileData": {"fileUri": source["url"]}})
                    elif block.get("type") == "tool_use":
                        parts.append({"functionCall": {"name": string_of(block.get("name")), "args": block["input"] if is_record(block.get("input")) else {}}})
                    elif block.get("type") == "tool_result":
                        parts.append({"functionResponse": {"name": string_of(block.get("tool_use_id")), "response": {"result": tool_result_text(block.get("content"))}}})
            contents.append((role, parts))
        if isinstance(body.get("tools"), list):
            for raw in body["tools"]:
                if is_record(raw) and isinstance(raw.get("name"), str):
                    declarations.append({
                        "name": raw["name"],
                        "description": string_of(raw.get("description")),
                        "parameters": raw["input_schema"] if is_record(raw.get("input_schema")) else {"type": "object", "properties": {}},
                    })
        if number_or_none(body.get("max_tokens")) is not None:
            config["maxOutputTokens"] = body["max_tokens"]
        stops = string_list(body.get("stop_sequences"))
        if stops:
            config["stopSequences"] = stops
    if number_or_none(body.get("temperature")) is not None:
        config["temperature"] = body["temperature"]
    top = body.get("top_p") if client_wire == "openai" else body.get("top_p")
    if number_or_none(top) is not None:
        config["topP"] = top
    out = {}
    system_parts = [c[1] for c in contents if c[0] == "system"]
    if system_parts:
        out["systemInstruction"] = {"parts": [{"text": system_parts[0]}]}
    out["contents"] = [{"role": role, "parts": parts} for role, parts in contents if role != "system"]
    if declarations:
        out["tools"] = [{"functionDeclarations": declarations}]
    if config:
        out["generationConfig"] = config
    return out


# --- provider answers, normalized: {text, calls, stop, usage} --------------

def stop_to_openai(reason):
    return {
        "end_turn": "stop", "max_tokens": "length", "tool_use": "tool_calls",
        "refusal": "content_filter", "STOP": "stop", "MAX_TOKENS": "length",
        "SAFETY": "content_filter", "stop": "stop", "length": "length",
        "tool_calls": "tool_calls", "function_call": "tool_calls",
        "content_filter": "content_filter",
    }.get(reason, "stop")


def stop_to_anthropic(reason):
    return {
        "stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use",
        "function_call": "tool_use", "content_filter": "refusal",
        "STOP": "end_turn", "MAX_TOKENS": "max_tokens", "SAFETY": "refusal",
        "end_turn": "end_turn", "max_tokens": "max_tokens",
        "tool_use": "tool_use", "refusal": "refusal",
    }.get(reason, "end_turn")


def parse_openai_json(body):
    choices = body.get("choices") if isinstance(body.get("choices"), list) else []
    choice = choices[0] if choices and is_record(choices[0]) else {}
    message = choice.get("message") if is_record(choice.get("message")) else {}
    content = message.get("content")
    text = content if isinstance(content, str) else blocks_text(content)
    calls = []
    raw_calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
    for call in raw_calls:
        if is_record(call) and is_record(call.get("function")):
            calls.append({
                "id": string_of(call.get("id")),
                "name": string_of(call["function"].get("name")),
                "arguments": call["function"]["arguments"] if isinstance(call["function"].get("arguments"), str) else dumps(arguments_of(call["function"].get("arguments"))),
            })
    usage = body.get("usage") if is_record(body.get("usage")) else {}
    return {
        "text": text,
        "calls": calls,
        "stop": string_of(choice.get("finish_reason")) or "stop",
        "usage": {
            "input": number_or_none(usage.get("prompt_tokens")) or 0,
            "output": number_or_none(usage.get("completion_tokens")) or 0,
        },
    }


def parse_anthropic_json(body):
    text_parts = []
    calls = []
    content = body.get("content") if isinstance(body.get("content"), list) else []
    for block in content:
        if not is_record(block):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            text_parts.append(block["text"])
        elif block.get("type") == "tool_use":
            calls.append({
                "id": string_of(block.get("id")),
                "name": string_of(block.get("name")),
                "arguments": dumps(block["input"]) if is_record(block.get("input")) else "{}",
            })
    usage = body.get("usage") if is_record(body.get("usage")) else {}
    return {
        "text": "".join(text_parts),
        "calls": calls,
        "stop": string_of(body.get("stop_reason")) or "end_turn",
        "usage": {
            "input": number_or_none(usage.get("input_tokens")) or 0,
            "output": number_or_none(usage.get("output_tokens")) or 0,
        },
    }


def parse_gemini_json(body):
    candidates = body.get("candidates") if isinstance(body.get("candidates"), list) else []
    candidate = candidates[0] if candidates and is_record(candidates[0]) else {}
    content = candidate.get("content") if is_record(candidate.get("content")) else {}
    text_parts = []
    calls = []
    parts = content.get("parts") if isinstance(content.get("parts"), list) else []
    for part in parts:
        if not is_record(part):
            continue
        if isinstance(part.get("text"), str):
            text_parts.append(part["text"])
        elif is_record(part.get("functionCall")):
            fn = part["functionCall"]
            calls.append({
                "id": "",
                "name": string_of(fn.get("name")),
                "arguments": dumps(fn["args"]) if is_record(fn.get("args")) else "{}",
            })
    usage = body.get("usageMetadata") if is_record(body.get("usageMetadata")) else {}
    return {
        "text": "".join(text_parts),
        "calls": calls,
        "stop": string_of(candidate.get("finishReason")) or "STOP",
        "usage": {
            "input": number_or_none(usage.get("promptTokenCount")) or 0,
            "output": number_or_none(usage.get("candidatesTokenCount")) or 0,
        },
    }


def parse_sse_frames(text):
    frames = []
    normalized = text.replace("\r\n", "\n")
    for frame in normalized.split("\n\n"):
        if frame.strip() == "":
            continue
        event = None
        data = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data.append(line[len("data:"):].lstrip(" ") if line.startswith("data: ") else line[len("data:"):])
        if data:
            frames.append((event, "\n".join(data)))
    return frames


def parse_openai_sse(text):
    text_parts = []
    calls = {}
    finish = None
    usage = {"input": 0, "output": 0}
    for _event, data in parse_sse_frames(text):
        payload = data.strip()
        if payload == "[DONE]":
            continue
        try:
            chunk = json.loads(payload)
        except ValueError:
            continue
        if not is_record(chunk):
            continue
        if is_record(chunk.get("usage")):
            usage = {
                "input": number_or_none(chunk["usage"].get("prompt_tokens")) or 0,
                "output": number_or_none(chunk["usage"].get("completion_tokens")) or 0,
            }
        choices = chunk.get("choices") if isinstance(chunk.get("choices"), list) else []
        if not choices or not is_record(choices[0]):
            continue
        choice = choices[0]
        delta = choice.get("delta") if is_record(choice.get("delta")) else {}
        if isinstance(delta.get("content"), str):
            text_parts.append(delta["content"])
        raw_calls = delta.get("tool_calls") if isinstance(delta.get("tool_calls"), list) else []
        for call in raw_calls:
            if not is_record(call):
                continue
            index = call.get("index") if isinstance(call.get("index"), int) else len(calls)
            entry = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
            if isinstance(call.get("id"), str) and call["id"] != "":
                entry["id"] = call["id"]
            if is_record(call.get("function")):
                if isinstance(call["function"].get("name"), str):
                    entry["name"] += call["function"]["name"]
                if isinstance(call["function"].get("arguments"), str):
                    entry["arguments"] += call["function"]["arguments"]
        if isinstance(choice.get("finish_reason"), str):
            finish = choice["finish_reason"]
    return {
        "text": "".join(text_parts),
        "calls": [calls[k] for k in sorted(calls)],
        "stop": finish or "stop",
        "usage": usage,
    }


def parse_anthropic_sse(text):
    text_parts = []
    blocks = {}
    stop = None
    usage = {"input": 0, "output": 0}
    for event, data in parse_sse_frames(text):
        try:
            payload = json.loads(data)
        except ValueError:
            continue
        if not is_record(payload):
            continue
        if event == "message_start":
            message = payload.get("message") if is_record(payload.get("message")) else {}
            mu = message.get("usage") if is_record(message.get("usage")) else {}
            usage["input"] = number_or_none(mu.get("input_tokens")) or 0
        elif event == "content_block_start":
            index = payload.get("index") if isinstance(payload.get("index"), int) else 0
            block = payload.get("content_block") if is_record(payload.get("content_block")) else {}
            blocks[index] = {"type": block.get("type"), "id": string_of(block.get("id")), "name": string_of(block.get("name")), "json": ""}
        elif event == "content_block_delta":
            index = payload.get("index") if isinstance(payload.get("index"), int) else 0
            delta = payload.get("delta") if is_record(payload.get("delta")) else {}
            if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                text_parts.append(delta["text"])
            elif delta.get("type") == "input_json_delta" and isinstance(delta.get("partial_json"), str):
                if index in blocks:
                    blocks[index]["json"] += delta["partial_json"]
        elif event == "message_delta":
            delta = payload.get("delta") if is_record(payload.get("delta")) else {}
            if isinstance(delta.get("stop_reason"), str):
                stop = delta["stop_reason"]
            mu = payload.get("usage") if is_record(payload.get("usage")) else {}
            if number_or_none(mu.get("output_tokens")) is not None:
                usage["output"] = mu["output_tokens"]
    calls = []
    for index in sorted(blocks):
        block = blocks[index]
        if block["type"] == "tool_use":
            calls.append({"id": block["id"], "name": block["name"], "arguments": block["json"] or "{}"})
    return {"text": "".join(text_parts), "calls": calls, "stop": stop or "end_turn", "usage": usage}


def parse_gemini_sse(text):
    text_parts = []
    calls = []
    finish = None
    usage = {"input": 0, "output": 0}
    for _event, data in parse_sse_frames(text):
        try:
            chunk = json.loads(data)
        except ValueError:
            continue
        if not is_record(chunk):
            continue
        candidates = chunk.get("candidates") if isinstance(chunk.get("candidates"), list) else []
        if candidates and is_record(candidates[0]):
            candidate = candidates[0]
            content = candidate.get("content") if is_record(candidate.get("content")) else {}
            parts = content.get("parts") if isinstance(content.get("parts"), list) else []
            for part in parts:
                if not is_record(part):
                    continue
                if isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
                elif is_record(part.get("functionCall")):
                    fn = part["functionCall"]
                    calls.append({
                        "id": "",
                        "name": string_of(fn.get("name")),
                        "arguments": dumps(fn["args"]) if is_record(fn.get("args")) else "{}",
                    })
            if isinstance(candidate.get("finishReason"), str):
                finish = candidate["finishReason"]
        if is_record(chunk.get("usageMetadata")):
            usage = {
                "input": number_or_none(chunk["usageMetadata"].get("promptTokenCount")) or 0,
                "output": number_or_none(chunk["usageMetadata"].get("candidatesTokenCount")) or 0,
            }
    return {"text": "".join(text_parts), "calls": calls, "stop": finish or "STOP", "usage": usage}


# --- client answers ----------------------------------------------------------

def build_openai_json(answer, model):
    message = {"role": "assistant", "content": answer["text"]}
    if answer["calls"]:
        message["tool_calls"] = [
            {
                "id": call["id"] or "call_%d" % i,
                "type": "function",
                "function": {"name": call["name"], "arguments": call["arguments"]},
            }
            for i, call in enumerate(answer["calls"])
        ]
    return {
        "id": "chatcmpl-aggregate",
        "object": "chat.completion",
        "created": FIXED_OPENAI_CREATED,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": stop_to_openai(answer["stop"])}],
        "usage": {
            "prompt_tokens": answer["usage"]["input"],
            "completion_tokens": answer["usage"]["output"],
            "total_tokens": answer["usage"]["input"] + answer["usage"]["output"],
        },
    }


def build_anthropic_json(answer, model):
    content = []
    if answer["text"] != "":
        content.append({"type": "text", "text": answer["text"]})
    for i, call in enumerate(answer["calls"]):
        content.append({
            "type": "tool_use",
            "id": call["id"] or "toolu_%d" % i,
            "name": call["name"],
            "input": arguments_of(call["arguments"]),
        })
    if not content:
        content.append({"type": "text", "text": ""})
    return {
        "id": "msg_aggregate",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_to_anthropic(answer["stop"]),
        "stop_sequence": None,
        "usage": {"input_tokens": answer["usage"]["input"], "output_tokens": answer["usage"]["output"]},
    }


def sse_frame(event, data_text):
    if event is None:
        return "data: %s\n\n" % data_text
    return "event: %s\ndata: %s\n\n" % (event, data_text)


def build_openai_sse(answer, model):
    frames = []
    base = {"id": "chatcmpl-aggregate", "object": "chat.completion.chunk", "created": FIXED_OPENAI_CREATED, "model": model}
    first = dict(base)
    first["choices"] = [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]
    frames.append(sse_frame(None, dumps(first)))
    if answer["text"] != "":
        chunk = dict(base)
        chunk["choices"] = [{"index": 0, "delta": {"content": answer["text"]}, "finish_reason": None}]
        frames.append(sse_frame(None, dumps(chunk)))
    for i, call in enumerate(answer["calls"]):
        chunk = dict(base)
        chunk["choices"] = [{
            "index": 0,
            "delta": {"tool_calls": [{
                "index": i,
                "id": call["id"] or "call_%d" % i,
                "type": "function",
                "function": {"name": call["name"], "arguments": call["arguments"]},
            }]},
            "finish_reason": None,
        }]
        frames.append(sse_frame(None, dumps(chunk)))
    finish = dict(base)
    finish["choices"] = [{"index": 0, "delta": {}, "finish_reason": stop_to_openai(answer["stop"])}]
    frames.append(sse_frame(None, dumps(finish)))
    usage_chunk = dict(base)
    usage_chunk["choices"] = []
    usage_chunk["usage"] = {
        "prompt_tokens": answer["usage"]["input"],
        "completion_tokens": answer["usage"]["output"],
        "total_tokens": answer["usage"]["input"] + answer["usage"]["output"],
    }
    frames.append(sse_frame(None, dumps(usage_chunk)))
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def build_anthropic_sse(answer, model):
    frames = []
    frames.append(sse_frame("message_start", dumps({
        "type": "message_start",
        "message": {
            "id": "msg_aggregate",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": answer["usage"]["input"], "output_tokens": 0},
        },
    })))
    index = 0
    if answer["text"] != "":
        frames.append(sse_frame("content_block_start", dumps({
            "type": "content_block_start", "index": index,
            "content_block": {"type": "text", "text": ""},
        })))
        frames.append(sse_frame("content_block_delta", dumps({
            "type": "content_block_delta", "index": index,
            "delta": {"type": "text_delta", "text": answer["text"]},
        })))
        frames.append(sse_frame("content_block_stop", dumps({"type": "content_block_stop", "index": index})))
        index += 1
    for i, call in enumerate(answer["calls"]):
        frames.append(sse_frame("content_block_start", dumps({
            "type": "content_block_start", "index": index,
            "content_block": {"type": "tool_use", "id": call["id"] or "toolu_%d" % i, "name": call["name"], "input": {}},
        })))
        if call["arguments"] != "":
            frames.append(sse_frame("content_block_delta", dumps({
                "type": "content_block_delta", "index": index,
                "delta": {"type": "input_json_delta", "partial_json": call["arguments"]},
            })))
        frames.append(sse_frame("content_block_stop", dumps({"type": "content_block_stop", "index": index})))
        index += 1
    frames.append(sse_frame("message_delta", dumps({
        "type": "message_delta",
        "delta": {"stop_reason": stop_to_anthropic(answer["stop"]), "stop_sequence": None},
        "usage": {"output_tokens": answer["usage"]["output"]},
    })))
    frames.append(sse_frame("message_stop", dumps({"type": "message_stop"})))
    return "".join(frames)


def openai_error(message, etype, code=None):
    return {"error": {"message": message, "type": etype, "code": code}}


def anthropic_error(etype, message):
    return {"type": "error", "error": {"type": etype, "message": message}}


def openai_model_list(models):
    return {
        "object": "list",
        "data": [{"id": m["id"], "object": "model", "created": FIXED_OPENAI_CREATED, "owned_by": m["provider"]} for m in models],
    }


def anthropic_model_list(models):
    return {
        "data": [{"type": "model", "id": m["id"], "display_name": m["display_name"], "created_at": FIXED_ANTHROPIC_CREATED_AT} for m in models],
        "has_more": False,
    }


# --------------------------------------------------------------------------
# Routing: best free model first, immediate failover on a refusal, and a
# provider that refuses twice drops behind its peers for the rest of the
# trial. Upstream calls are serialized under one lock — a provider caps
# concurrent calls, and a deterministic order keeps replays identical.
# --------------------------------------------------------------------------

def body_requirements(client_wire, body):
    """Requirement fallback read from the body itself (used when the brief's
    manifest is exhausted): tools and images and JSON mode are visible in the
    body; context length is not, so none is claimed."""
    tools = isinstance(body.get("tools"), list) and len(body["tools"]) > 0
    vision = False
    if client_wire == "openai":
        messages = body.get("messages") if isinstance(body.get("messages"), list) else []
        for message in messages:
            content = message.get("content") if is_record(message) else None
            if isinstance(content, list):
                for part in content:
                    if is_record(part) and part.get("type") == "image_url":
                        vision = True
        fmt = body.get("response_format")
        json_mode = is_record(fmt) and fmt.get("type") in ("json_object", "json_schema")
    else:
        messages = body.get("messages") if isinstance(body.get("messages"), list) else []
        for message in messages:
            content = message.get("content") if is_record(message) else None
            if isinstance(content, list):
                for part in content:
                    if is_record(part) and part.get("type") == "image":
                        vision = True
        json_mode = False
    return {"tools": tools, "vision": vision, "json": json_mode}


def wire_allowed(client_wire, body, provider_wire):
    """A JSON-mode flag an anthropic-wire provider cannot echo: an OpenAI
    response_format request may not be routed to an anthropic-wire provider
    (its echo would report json_mode false). Gemini takes the flag via
    generationConfig, so it stays legal."""
    if provider_wire == "anthropic" and client_wire == "openai":
        fmt = body.get("response_format")
        if is_record(fmt) and fmt.get("type") in ("json_object", "json_schema"):
            return False
    return True


class Router:
    def __init__(self, brief, probes):
        self.brief = brief
        self.probes = {p["provider"]["id"]: p for p in probes if p["ok"]}
        self.directory = {p["id"]: p for p in brief.providers}
        self.failures = {}
        self.counted = {}
        self.arrival = 0
        self.lock = threading.Lock()

    def requirements_for(self, client_wire, body):
        manifest = self.brief.workload
        if manifest is not None and self.arrival < len(manifest):
            row = manifest[self.arrival]
            self.arrival += 1
            return dict(row["requires"])
        self.arrival += 1
        return body_requirements(client_wire, body)

    def candidates(self, requires, client_wire, body):
        out = []
        for pid, probe in self.probes.items():
            provider = self.directory[pid]
            for model in probe["models"]:
                row = rating_row(self.brief, pid, model["id"])
                if not model_free(model, row):
                    continue
                caps = capabilities_of(model, row)
                if not meets_requirements(requires, caps):
                    continue
                if not wire_allowed(client_wire, body, provider["wire"]):
                    continue
                out.append({
                    "provider": provider,
                    "model": model["id"],
                    "rating": row["rating_milli"] if row is not None else -1,
                    "caps": caps,
                })
        demoted = {pid for pid, n in self.failures.items() if n >= 2}

        def sort_key(c):
            return (0 if c["provider"]["id"] in demoted else 1, -c["rating"], self.counted.get(c["provider"]["id"], 0), c["provider"]["id"], c["model"])

        out.sort(key=sort_key)
        return out

    def upstream(self, candidate, client_wire, body, stream):
        """One upstream call. Returns (status, answer-or-None)."""
        provider = candidate["provider"]
        model = candidate["model"]
        base = join_base(ENV_URL, provider["base"])
        headers = {"accept": "application/json"}
        headers.update(auth_headers(provider["auth"], provider["key"]))
        if provider["wire"] == "openai":
            payload = to_openai_body(client_wire, body, model)
            url = base + "/v1/chat/completions"
            headers["content-type"] = "application/json"
        elif provider["wire"] == "anthropic":
            payload = to_anthropic_body(client_wire, body, model)
            url = base + "/v1/messages"
            headers["content-type"] = "application/json"
            headers["anthropic-version"] = ANTHROPIC_VERSION
        else:
            payload = to_gemini_body(client_wire, body)
            verb = "streamGenerateContent" if stream else "generateContent"
            url = base + "/v1beta/models/%s:%s" % (model, verb)
            headers["content-type"] = "application/json"
        raw = dumps(payload).encode("utf-8")
        try:
            status, _headers, data = http_request("POST", url, headers, raw, UPSTREAM_TIMEOUT_S)
        except Exception as err:
            log("upstream %s/%s failed to connect: %s" % (provider["id"], model, err))
            return 502, None
        if status < 200 or status >= 300:
            return status, None
        text = data.decode("utf-8", "replace")
        try:
            if provider["wire"] == "openai":
                answer = parse_openai_sse(text) if stream else parse_openai_json(json.loads(text))
            elif provider["wire"] == "anthropic":
                answer = parse_anthropic_sse(text) if stream else parse_anthropic_json(json.loads(text))
            else:
                answer = parse_gemini_sse(text) if stream else parse_gemini_json(json.loads(text))
        except Exception as err:
            log("upstream %s/%s answer unreadable: %s" % (provider["id"], model, err))
            return 502, None
        answer["model"] = model
        return status, answer

    def serve(self, client_wire, body):
        """(status, answer-or-None) for one client request."""
        stream = body.get("stream") is True
        requires = self.requirements_for(client_wire, body)
        blocked = set()
        with self.lock:
            for candidate in self.candidates(requires, client_wire, body):
                pid = candidate["provider"]["id"]
                if pid in blocked:
                    continue
                status, answer = self.upstream(candidate, client_wire, body, stream)
                if answer is not None:
                    self.counted[pid] = self.counted.get(pid, 0) + 1
                    return 200, answer
                if status in (400, 401, 403, 404):
                    # The provider did not serve this model under this key;
                    # the call spends no budget, so another model on the same
                    # provider stays eligible. Try the next candidate.
                    continue
                # A counted refusal (402/429/5xx or a broken answer): the
                # provider is done for this request; fail over.
                blocked.add(pid)
                self.failures[pid] = self.failures.get(pid, 0) + 1
                self.counted[pid] = self.counted.get(pid, 0) + 1
        return 503, None


# --------------------------------------------------------------------------
# The HTTP server and the trial lifecycle.
# --------------------------------------------------------------------------

STATE = {"router": None, "catalog": None, "free_models": []}


def make_handler():
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _send(self, status, payload, content_type):
            body = payload if isinstance(payload, bytes) else payload.encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, status, value):
            self._send(status, dumps(value), "application/json")

        def _read_json(self):
            length = int(self.headers.get("content-length") or 0)
            if length > MAX_BODY_BYTES:
                return None
            raw = self.rfile.read(length) if length > 0 else b""
            try:
                value = json.loads(raw.decode("utf-8")) if raw else {}
            except ValueError:
                return None
            return value if is_record(value) else None

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/v1/models":
                self._json(200, openai_model_list(STATE["free_models"]))
                return
            if path == "/v1beta/models":
                data = [{
                    "name": "models/" + m["id"],
                    "displayName": m["display_name"],
                    "inputTokenLimit": m["context_length"],
                } for m in STATE["free_models"]]
                self._json(200, {"models": data})
                return
            self._json(404, openai_error("not found: %s" % path, "not_found_error"))

        def do_POST(self):
            path = urlparse(self.path).path
            if path == "/v1/models":
                # Anthropic model listing: POST with an x-api-key header and no
                # inference fields; real inference posts carry messages.
                body = self._read_json()
                if body is not None and "messages" not in body:
                    self._json(200, anthropic_model_list(STATE["free_models"]))
                    return
                self._json(400, anthropic_error("invalid_request_error", "missing messages"))
                return
            if path == "/v1/chat/completions":
                self._inference("openai")
                return
            if path == "/v1/messages":
                self._inference("anthropic")
                return
            self._json(404, openai_error("not found: %s" % path, "not_found_error"))

        def _inference(self, client_wire):
            body = self._read_json()
            if body is None:
                if client_wire == "openai":
                    self._json(400, openai_error("the request body is not a JSON object", "invalid_request_error"))
                else:
                    self._json(400, anthropic_error("invalid_request_error", "the request body is not a JSON object"))
                return
            stream = body.get("stream") is True
            status, answer = STATE["router"].serve(client_wire, body)
            if answer is None:
                message = "no free provider could serve this request"
                if client_wire == "openai":
                    self._json(503, openai_error(message, "api_error"))
                else:
                    self._json(503, anthropic_error("api_error", message))
                return
            model = answer["model"]
            if client_wire == "openai":
                if stream:
                    self._send(200, build_openai_sse(answer, model), "text/event-stream")
                else:
                    self._json(200, build_openai_json(answer, model))
            else:
                if stream:
                    self._send(200, build_anthropic_sse(answer, model), "text/event-stream")
                else:
                    self._json(200, build_anthropic_json(answer, model))

    return Handler


def register_with_env(port, catalog):
    body = dumps({"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": catalog}).encode("utf-8")
    last_error = None
    for _attempt in range(50):
        try:
            status, _headers, data = http_request(
                "POST", ENV_URL + "/register",
                {"content-type": "application/json"}, body, 5.0,
            )
        except Exception as err:
            last_error = err
            threading.Event().wait(0.1)
            continue
        if status < 200 or status >= 300:
            raise RuntimeError("register answered HTTP %d: %s" % (status, data.decode("utf-8", "replace")[:200]))
        payload = json.loads(data.decode("utf-8"))
        if payload.get("ok") is not True:
            raise RuntimeError("register did not confirm: %s" % dumps(payload))
        log("registered on port %d; the client will send %s requests" % (port, payload.get("requests")))
        return
    raise RuntimeError("register never reached the env: %s" % last_error)


def watch_work():
    while True:
        try:
            status, _headers, data = http_request("GET", ENV_URL + "/work", {}, None, 2.0)
            if status >= 200 and status < 300:
                payload = json.loads(data.decode("utf-8"))
                if payload.get("done") is True:
                    log("the client is done; exiting")
                    os._exit(0)
        except Exception:
            pass
        threading.Event().wait(POLL_INTERVAL_S)


def free_model_rows(catalog, brief, probes):
    rows = []
    seen = set()
    by_id = {p["provider"]["id"]: p for p in probes}
    for entry in catalog["providers"]:
        probe = by_id.get(entry["id"])
        display = {}
        if probe is not None:
            display = {m["id"]: m for m in probe["models"]}
        for model in entry["models"]:
            if model["free"] is not True or model["id"] in seen:
                continue
            seen.add(model["id"])
            listed = display.get(model["id"])
            rows.append({
                "id": model["id"],
                "provider": entry["id"],
                "display_name": listed["display_name"] if listed is not None else model["id"],
                "context_length": model["context_length"],
            })
    return rows


def main(argv):
    list_mode = "--list" in argv
    brief = read_brief(PROMPT_FILE)
    if brief.error is not None:
        if list_mode:
            sys.stdout.write(dumps({"providers": []}) + "\n")
            sys.stdout.flush()
            return 0
        log(brief.error)
        return 1
    if ENV_URL == "":
        if list_mode:
            sys.stdout.write(dumps({"providers": []}) + "\n")
            sys.stdout.flush()
            return 0
        log("MAKEFUN_ENV_URL is not set")
        return 1
    with ThreadPoolExecutor(max_workers=10) as pool:
        probes = list(pool.map(lambda p: probe_provider(ENV_URL, p), brief.providers))
    catalog = build_catalog(probes, brief)
    if list_mode:
        sys.stdout.write(dumps(catalog) + "\n")
        sys.stdout.flush()
        return 0
    STATE["router"] = Router(brief, probes)
    STATE["catalog"] = catalog
    STATE["free_models"] = free_model_rows(catalog, brief, probes)
    server = ThreadingHTTPServer(("0.0.0.0", 0), make_handler())
    server.daemon_threads = True
    port = server.server_address[1]
    if port == 8080:
        server.server_close()
        server = ThreadingHTTPServer(("0.0.0.0", 0), make_handler())
        server.daemon_threads = True
        port = server.server_address[1]
    log("listening on 0.0.0.0:%d" % port)
    register_with_env(port, catalog)
    threading.Thread(target=watch_work, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
