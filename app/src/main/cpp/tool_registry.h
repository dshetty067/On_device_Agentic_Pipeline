#pragma once
#include <string>
#include <vector>
#include <functional>
#include <map>
#include <unordered_map>

// ── Tool Parameter Schema ──────────────────────────────────────────────────
struct ToolParam {
    std::string name;
    std::string type;        // "string" | "number" | "boolean"
    std::string description;
    bool        required = true;
};

// ── Tool Definition ────────────────────────────────────────────────────────
struct ToolDefinition {
    std::string              name;
    std::string              description;
    std::vector<ToolParam>   params;
    // The actual executor: receives parsed args, returns result string
    std::function<std::string(const std::map<std::string, std::string>&)> executor;
};

// ── Parsed Tool Call (extracted from LLM output) ───────────────────────────
struct ToolCall {
    bool        valid     = false;
    std::string tool_name;
    std::map<std::string, std::string> args;
    std::string raw_json; // original JSON text from LLM output
};

// ── Tool Registry ──────────────────────────────────────────────────────────
class ToolRegistry {
public:
    void registerTool(ToolDefinition def) {
        tools_[def.name] = std::move(def);
    }

    const ToolDefinition* getTool(const std::string& name) const {
        auto it = tools_.find(name);
        return (it != tools_.end()) ? &it->second : nullptr;
    }

    const std::unordered_map<std::string, ToolDefinition>& allTools() const {
        return tools_;
    }

    // Serialize registry into a system prompt block (JSON-schema style)
    // This is injected into every LLM call transparently
    std::string buildSystemPromptBlock() const;

private:
    std::unordered_map<std::string, ToolDefinition> tools_;
};

// ── JSON Tool-Call Parser ──────────────────────────────────────────────────
// Scans LLM output for a JSON tool_call block
// Returns a parsed ToolCall (valid=false if none found)
ToolCall parseToolCall(const std::string& llm_output);

// ── Global registry accessor ───────────────────────────────────────────────
ToolRegistry& getToolRegistry();