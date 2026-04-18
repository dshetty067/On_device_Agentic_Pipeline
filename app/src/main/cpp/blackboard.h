#pragma once
#include <string>
#include <map>

struct AgentBlackboard {
    // Input
    std::string user_query;

    // Intent classification result (from IntentNode)
    std::string planned_action;     // "use_tool" | "general_answer"
    std::string selected_tool;      // "weather" | "rag" | "book_flight" | ""

    // Tool result
    std::string tool_result;
    bool        tool_succeeded = false;

    // Final answer streamed to UI
    std::string final_answer;

    void reset() {
        planned_action  = "";
        selected_tool   = "";
        tool_result     = "";
        tool_succeeded  = false;
        final_answer    = "";
    }
};