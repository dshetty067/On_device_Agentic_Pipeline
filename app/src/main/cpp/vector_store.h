#pragma once
#include <string>
#include <vector>

struct SearchResult {
    int   chunk_index;
    float score;
    std::string text;
};

void vector_store_build(
        const std::vector<std::vector<float>>& embeddings,
        const std::vector<std::string>& chunks
);

// Save index + chunks to internal storage dir
void vector_store_save(const std::string& dir);

// Load index + chunks from internal storage dir
bool vector_store_load(const std::string& dir);

// Top-k search using hnswlib
std::vector<SearchResult> vector_store_search(
        const std::vector<float>& query_emb,
        int top_k = 3
);

float cosine_similarity(
        const std::vector<float>& a,
        const std::vector<float>& b
);

std::vector<SearchResult> rerank_results(
        const std::vector<float>& query_emb,
        std::vector<SearchResult>& results,
        const std::vector<std::vector<float>>& stored_embeddings
);

bool vector_store_is_ready();