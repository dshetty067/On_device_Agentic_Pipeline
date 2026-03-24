#pragma once
#include <string>
#include <functional>
#include "llama.h"

void load_model(const std::string& path);

std::string generate(
        const std::string& prompt,
        std::function<void(const std::string&)> on_token = nullptr
);

const llama_vocab* get_vocab();
llama_context*     get_ctx();