#pragma once
#include <string>
#include <vector>
#include "llama.h"

void load_embedding_model(const std::string& path);
void unload_embedding_model();

// Base embedding (internal use)
std::vector<float> embed_text(const std::string& text);

// 🔥 ADD THESE (VERY IMPORTANT)
std::vector<float> embed_query(const std::string& query);
std::vector<float> embed_passage(const std::string& passage);

// Return embedding dimension
int get_embedding_dim();