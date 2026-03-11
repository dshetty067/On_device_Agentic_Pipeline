#pragma once
#include <string>
#include <functional>

struct AgentState {
    std::string pdf_path;
    std::string store_dir;
    std::string query;
    std::string retrieved_context;
    std::string response;
};

void run_agent(
        const std::string& pdf_path,
        const std::string& store_dir,
        const std::string& prompt,
        std::function<void(const std::string&)> on_token,
        std::function<void(const std::string&)> on_status
);