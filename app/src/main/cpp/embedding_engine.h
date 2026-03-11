#pragma once
#include <string>
#include <vector>
#include "llama.h"

// Embedding model (separate from generation model)
void load_embedding_model(const std::string& path);
void unload_embedding_model();

// Returns a flat float vector of dimension embedding_dim
std::vector<float> embed_text(const std::string& text);

int get_embedding_dim();