#include "vector_store.h"
#include "hnswlib/hnswlib.h"

#include <fstream>
#include <android/log.h>

#define LOG_TAG "VectorStore"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Use InnerProduct space — correct for cosine sim on L2-normalized vectors
static hnswlib::InnerProductSpace*          g_space = nullptr;
static hnswlib::HierarchicalNSW<float>*     g_index = nullptr;
static std::vector<std::string>             g_chunks;
static int                                  g_dim   = 0;
static bool                                 g_ready = false;

// ── helpers ────────────────────────────────────────────────────────────────

static void cleanup() {
    delete g_index; g_index = nullptr;
    delete g_space; g_space = nullptr;
    g_ready = false;
}

// ── build ──────────────────────────────────────────────────────────────────

void vector_store_build(
        const std::vector<std::vector<float>>& embeddings,
        const std::vector<std::string>& chunks)
{
    if (embeddings.empty()) {
        LOGE("vector_store_build: no embeddings");
        return;
    }

    cleanup();

    g_dim    = (int)embeddings[0].size();
    g_chunks = chunks;

    g_space = new hnswlib::InnerProductSpace(g_dim);

    // M=16, ef_construction=200 — good defaults for mobile scale
    int M                = 16;
    int ef_construction  = 200;
    int max_elements     = (int)embeddings.size();

    g_index = new hnswlib::HierarchicalNSW<float>(
            g_space, max_elements, M, ef_construction
    );

    for (int i = 0; i < (int)embeddings.size(); i++) {
        g_index->addPoint(embeddings[i].data(), (hnswlib::labeltype)i);
        if (i % 10 == 0)
            LOGI("Indexed %d/%d", i+1, (int)embeddings.size());
    }

    // Set ef for search (higher = more accurate, slower)
    g_index->setEf(50);

    g_ready = true;
    LOGI("HNSW index built: %d vectors, dim=%d", max_elements, g_dim);
}

// ── save ───────────────────────────────────────────────────────────────────

void vector_store_save(const std::string& dir)
{
    if (!g_ready) { LOGE("save: index not ready"); return; }

    // Save HNSW index (hnswlib built-in serialization)
    std::string index_path = dir + "/hnsw_index.bin";
    g_index->saveIndex(index_path);
    LOGI("HNSW index saved: %s", index_path.c_str());

    // Save chunks + dim as a simple binary file
    std::string chunk_path = dir + "/chunks.bin";
    std::ofstream cf(chunk_path, std::ios::binary);
    if (!cf) { LOGE("Cannot write chunks.bin"); return; }

    cf.write((char*)&g_dim, sizeof(int));

    int n = (int)g_chunks.size();
    cf.write((char*)&n, sizeof(int));

    for (auto& c : g_chunks) {
        int len = (int)c.size();
        cf.write((char*)&len, sizeof(int));
        cf.write(c.data(), len);
    }
    cf.close();

    LOGI("Chunks saved: %s (%d chunks)", chunk_path.c_str(), n);
}

// ── load ───────────────────────────────────────────────────────────────────

bool vector_store_load(const std::string& dir)
{
    std::string index_path = dir + "/hnsw_index.bin";
    std::string chunk_path = dir + "/chunks.bin";

    // Check files exist
    std::ifstream check(index_path);
    if (!check.good()) {
        LOGI("No saved HNSW index found at %s", index_path.c_str());
        return false;
    }
    check.close();

    // Load chunks first to get g_dim
    std::ifstream cf(chunk_path, std::ios::binary);
    if (!cf) { LOGE("Cannot read chunks.bin"); return false; }

    cf.read((char*)&g_dim, sizeof(int));
    int n;
    cf.read((char*)&n, sizeof(int));
    g_chunks.resize(n);
    for (int i = 0; i < n; i++) {
        int len;
        cf.read((char*)&len, sizeof(int));
        g_chunks[i].resize(len);
        cf.read(&g_chunks[i][0], len);
    }
    cf.close();

    LOGI("Chunks loaded: %d chunks, dim=%d", n, g_dim);

    // Load HNSW index
    cleanup();
    g_space = new hnswlib::InnerProductSpace(g_dim);
    g_index = new hnswlib::HierarchicalNSW<float>(
            g_space, index_path, false, (size_t)n
    );
    g_index->setEf(50);

    g_ready = true;
    LOGI("HNSW index loaded from %s", index_path.c_str());
    return true;
}

// ── search ─────────────────────────────────────────────────────────────────

std::vector<SearchResult> vector_store_search(
        const std::vector<float>& query_emb,
        int top_k)
{
    if (!g_ready || !g_index) {
        LOGE("vector_store_search: index not ready");
        return {};
    }

    int k = std::min(top_k, (int)g_index->getCurrentElementCount());
    if (k == 0) return {};

    // hnswlib returns a priority queue of (dist, label)
    auto result = g_index->searchKnn(query_emb.data(), k);

    // Convert to vector (comes out closest-last, so reverse)
    std::vector<SearchResult> out;
    while (!result.empty()) {
        auto [dist, label] = result.top();
        result.pop();

        int idx = (int)label;
        // For InnerProduct space, dist = 1 - dot, so score = 1 - dist
        float score = 1.0f - dist;

        LOGI("SearchResult: chunk=%d score=%.4f", idx, score);

        out.push_back({
                              idx,
                              score,
                              (idx < (int)g_chunks.size()) ? g_chunks[idx] : ""
                      });
    }

    // Reverse so highest score is first
    std::reverse(out.begin(), out.end());
    return out;
}

bool vector_store_is_ready() { return g_ready; }