#include "tool_registry.h"
#include <sstream>
#include <android/log.h>

#define LOG_TAG "ToolRegistry"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// ── Global singleton ───────────────────────────────────────────────────────
ToolRegistry& getToolRegistry() {
    static ToolRegistry instance;
    return instance;
}

// ── Build system prompt block ──────────────────────────────────────────────
// Produces a JSON-schema-style block that the LLM reads as instructions.
// Format mirrors what LangGraph does silently — we do it explicitly.
//
// Example output injected into system prompt:
// ---
// You have access to the following tools. To use a tool, respond with ONLY
// a JSON object in this exact format (no other text):
// {"tool_call": {"name": "<tool_name>", "args": {"param": "value"}}}
//
// TOOLS:
// 1. search_pdf
//    Description: Search the PDF document for relevant passages
//    Parameters:
//      - query (string, required): The search query
// ---
std::string ToolRegistry::buildSystemPromptBlock() const {
    if (tools_.empty()) return "";

    std::ostringstream oss;
    oss << "You have access to the following tools.\n"
        << "When a tool is needed, respond with ONLY this JSON (no prose before or after):\n"
        << "{\"tool_call\": {\"name\": \"<tool_name>\", \"args\": {\"param\": \"value\"}}}\n\n"
        << "When NO tool is needed, respond normally in plain text.\n\n"
        << "AVAILABLE TOOLS:\n";

    int idx = 1;
    for (const auto& [name, def] : tools_) {
        oss << idx++ << ". " << def.name << "\n"
            << "   Description: " << def.description << "\n";
        if (!def.params.empty()) {
            oss << "   Parameters:\n";
            for (const auto& p : def.params) {
                oss << "     - " << p.name
                    << " (" << p.type
                    << (p.required ? ", required" : ", optional")
                    << "): " << p.description << "\n";
            }
        }
        oss << "\n";
    }
    return oss.str();
}

// ── Minimal JSON parser helpers ────────────────────────────────────────────
// We only need to parse {"tool_call": {"name": "...", "args": {...}}}
// Using a hand-rolled parser avoids adding nlohmann/json dependency on Android.

static std::string extractJsonString(const std::string& src, size_t& pos) {
    // pos points to opening '"'
    std::string result;
    ++pos; // skip "
    while (pos < src.size() && src[pos] != '"') {
        if (src[pos] == '\\' && pos + 1 < src.size()) {
            ++pos;
            switch (src[pos]) {
                case 'n':  result += '\n'; break;
                case 't':  result += '\t'; break;
                case '"':  result += '"';  break;
                case '\\': result += '\\';break;
                default:   result += src[pos]; break;
            }
        } else {
            result += src[pos];
        }
        ++pos;
    }
    ++pos; // skip closing "
    return result;
}

static void skipWhitespace(const std::string& s, size_t& pos) {
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\n' ||
                              s[pos] == '\r' || s[pos] == '\t')) {
        ++pos;
    }
}

// ── parseToolCall ──────────────────────────────────────────────────────────
// Scans the LLM output for a JSON block containing "tool_call".
// Returns ToolCall with valid=true only if parsing succeeds completely.
ToolCall parseToolCall(const std::string& llm_output) {
    ToolCall result;

    // Find the outermost { that contains "tool_call"
    size_t json_start = std::string::npos;
    size_t search_pos = 0;
    while ((search_pos = llm_output.find('{', search_pos)) != std::string::npos) {
        // Check if "tool_call" appears in reasonable proximity
        size_t lookahead = llm_output.find("tool_call", search_pos);
        if (lookahead != std::string::npos && lookahead - search_pos < 50) {
            json_start = search_pos;
            break;
        }
        ++search_pos;
    }

    if (json_start == std::string::npos) {
        LOGI("parseToolCall: no tool_call JSON found in output");
        return result; // valid=false
    }

    // Find matching closing brace (simple depth-tracking)
    int depth = 0;
    size_t json_end = json_start;
    for (; json_end < llm_output.size(); ++json_end) {
        if (llm_output[json_end] == '{') ++depth;
        else if (llm_output[json_end] == '}') {
            --depth;
            if (depth == 0) { ++json_end; break; }
        }
    }

    result.raw_json = llm_output.substr(json_start, json_end - json_start);
    LOGI("parseToolCall: extracted JSON: %s", result.raw_json.c_str());

    // State-machine parse: find "name" and "args" inside "tool_call"
    const std::string& j = result.raw_json;
    size_t pos = 0;

    // Find "name":
    size_t name_pos = j.find("\"name\"");
    if (name_pos == std::string::npos) {
        LOGE("parseToolCall: missing 'name' key");
        return result;
    }
    pos = name_pos + 6; // skip "name"
    skipWhitespace(j, pos);
    if (pos >= j.size() || j[pos] != ':') return result;
    ++pos;
    skipWhitespace(j, pos);
    if (pos >= j.size() || j[pos] != '"') return result;
    result.tool_name = extractJsonString(j, pos);

    // Find "args":
    size_t args_pos = j.find("\"args\"");
    if (args_pos == std::string::npos) {
        // No args is valid for zero-parameter tools
        result.valid = !result.tool_name.empty();
        return result;
    }
    pos = args_pos + 6;
    skipWhitespace(j, pos);
    if (pos >= j.size() || j[pos] != ':') return result;
    ++pos;
    skipWhitespace(j, pos);
    if (pos >= j.size() || j[pos] != '{') return result;
    ++pos; // enter args object

    // Parse key-value pairs inside args
    while (pos < j.size()) {
        skipWhitespace(j, pos);
        if (pos >= j.size() || j[pos] == '}') break;
        if (j[pos] == ',') { ++pos; continue; }
        if (j[pos] != '"') { ++pos; continue; } // skip unexpected chars

        std::string key = extractJsonString(j, pos);
        skipWhitespace(j, pos);
        if (pos >= j.size() || j[pos] != ':') break;
        ++pos;
        skipWhitespace(j, pos);

        std::string val;
        if (pos < j.size() && j[pos] == '"') {
            val = extractJsonString(j, pos);
        } else {
            // Non-string value: read until , or }
            size_t val_start = pos;
            while (pos < j.size() && j[pos] != ',' && j[pos] != '}') ++pos;
            val = j.substr(val_start, pos - val_start);
            // trim
            while (!val.empty() && (val.back() == ' ' || val.back() == '\n')) val.pop_back();
        }
        result.args[key] = val;
        LOGI("parseToolCall: arg[%s] = %s", key.c_str(), val.c_str());
    }

    result.valid = !result.tool_name.empty();
    LOGI("parseToolCall: valid=%d tool=%s args_count=%zu",
         result.valid, result.tool_name.c_str(), result.args.size());
    return result;
}