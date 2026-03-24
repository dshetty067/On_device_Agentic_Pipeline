#pragma once
#include <string>
#include <vector>
#include "llama.h"

void load_embedding_model(const std::string& path);
void unload_embedding_model();

std::vector<float> embed_text(const std::string& text);

int get_embedding_dim();