#pragma once
#include <string>
#include <functional>

void load_model(const std::string& path);
std::string generate(const std::string& prompt, std::function<void(const std::string&)> on_token = nullptr);