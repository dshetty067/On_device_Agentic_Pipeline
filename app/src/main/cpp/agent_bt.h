#pragma once
#include <string>
#include <functional>
#include "blackboard.h"

// ── Public API ─────────────────────────────────────────────────────────────

// Initialises the tool registry with default tools (search_pdf, calculate, etc.)
// Call once at startup AFTER models are loaded.
void init_tool_registry();

// Run the full BT agent for one user turn.
// on_token  : called for each generated token (streaming)
// on_status : called with pipeline status messages for the UI ticker
void run_agent(
        const std::string& pdf_path,
        const std::string& store_dir,
        const std::string& query,
        std::function<void(const std::string&)> on_token,
        std::function<void(const std::string&)> on_status
);

// Access the live blackboard (for JNI inspection if needed)
const AgentBlackboard& get_blackboard();