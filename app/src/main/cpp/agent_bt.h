#pragma once
#include <string>
#include <functional>

void run_agent(
        const std::string& query,
        std::function<void(const std::string&)> on_token,
        std::function<void(const std::string&)> on_status
);

void agent_set_pdf_path(const std::string& path);
void agent_set_index_dir(const std::string& dir);