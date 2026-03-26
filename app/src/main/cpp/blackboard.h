#pragma once
#include <string>
#include <vector>
#include <map>
#include "tool_registry.h"

// ── Message role for history ───────────────────────────────────────────────
enum class MsgRole { SYSTEM, USER, ASSISTANT, TOOL };

struct HistoryMessage {
    MsgRole     role;
    std::string content;
    std::string tool_name; // only for TOOL role
};

// ── Relevance classification ───────────────────────────────────────────────
enum class RelevanceLevel { NONE, LOW, MEDIUM, HIGH };

// ── Agent Blackboard ───────────────────────────────────────────────────────
// Single shared-state object passed through the entire BT.
// Named AgentBlackboard to avoid collision with BT::Blackboard.
// This replaces the old static `AgentState`.
struct AgentBlackboard {
    // ── Input ──────────────────────────────────────────────────────────────
    std::string pdf_path;
    std::string store_dir;
    std::string user_query;       // raw user input this turn

    // ── RAG retrieval ──────────────────────────────────────────────────────
    std::string retrieved_context;   // top-N chunks joined
    float       best_relevance_score = 0.0f; // cosine similarity of top hit
    RelevanceLevel relevance_level   = RelevanceLevel::NONE;

    // ── Planner output ─────────────────────────────────────────────────────
    std::string planned_action;      // "use_tool" | "rag_answer" | "general_answer"
    std::string planner_reasoning;   // chain-of-thought from planner node

    // ── Tool call state ────────────────────────────────────────────────────
    ToolCall    pending_tool_call;   // parsed from LLM output
    std::string tool_result;         // executor output
    bool        tool_succeeded = false;
    int         tool_call_count = 0; // guard against infinite loops
    static constexpr int MAX_TOOL_CALLS = 3;

    // ── LLM output ─────────────────────────────────────────────────────────
    std::string last_llm_output;     // raw output of latest LLM call
    std::string final_answer;        // cleaned, validated answer for UI

    // ── Message history (multi-turn context window) ─────────────────────
    std::vector<HistoryMessage> message_history;

    // ── Validation ─────────────────────────────────────────────────────────
    bool        answer_validated  = false;
    std::string validation_reason; // why answer was accepted/rejected

    // ── Pipeline flags ──────────────────────────────────────────────────────
    bool        embeddings_ready = false;
    bool        pdf_loaded       = false;

    // ── Helpers ────────────────────────────────────────────────────────────
    void appendHistory(MsgRole role, const std::string& content,
                       const std::string& tool_name = "") {
        message_history.push_back({role, content, tool_name});
        // Keep history bounded: drop oldest non-system messages if too long
        constexpr size_t MAX_HISTORY = 12;
        while (message_history.size() > MAX_HISTORY) {
            // Never drop system message (index 0)
            if (message_history.size() > 1)
                message_history.erase(message_history.begin() + 1);
            else
                break;
        }
    }

    void reset_per_turn() {
        retrieved_context    = "";
        best_relevance_score = 0.0f;
        relevance_level      = RelevanceLevel::NONE;
        planned_action       = "";
        planner_reasoning    = "";
        pending_tool_call    = ToolCall{};
        tool_result          = "";
        tool_succeeded       = false;
        tool_call_count      = 0;
        last_llm_output      = "";
        final_answer         = "";
        answer_validated     = false;
        validation_reason    = "";
    }
};