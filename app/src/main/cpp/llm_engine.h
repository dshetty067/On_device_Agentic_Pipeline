#pragma once
#include <string>
#include <functional>

void load_model(const std::string& path);

std::string classify_intent(const std::string& query);

std::string general_answer(
        const std::string& query,
        std::function<void(const std::string&)> on_token = nullptr
);

std::string refined_answer(
        const std::string& query,
        const std::string& tool_name,
        const std::string& tool_result,
        std::function<void(const std::string&)> on_token = nullptr
);