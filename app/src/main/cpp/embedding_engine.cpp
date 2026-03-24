#include "embedding_engine.h"
#include <android/log.h>
#include <cmath>
#include <cstring>
#include <thread>

#define LOG_TAG "EmbeddingEngine"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static llama_model*   emb_model = nullptr;
static llama_context* emb_ctx   = nullptr;
static const llama_vocab* emb_vocab = nullptr;
static int emb_dim = 0;

void load_embedding_model(const std::string& path)
{
    LOGI("load_embedding_model: %s", path.c_str());

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    emb_model = llama_model_load_from_file(path.c_str(), mparams);
    if (!emb_model) {
        LOGE("Failed to load embedding model");
        return;
    }

    llama_context_params cparams = llama_context_default_params();

    cparams.n_ctx        = 256;
    cparams.n_batch      = 128;
    cparams.n_threads    = std::thread::hardware_concurrency();
    cparams.embeddings   = true;
    cparams.pooling_type = LLAMA_POOLING_TYPE_MEAN;

    emb_ctx = llama_init_from_model(emb_model, cparams);
    if (!emb_ctx) {
        LOGE("Failed to create embedding context");
        return;
    }

    emb_vocab = llama_model_get_vocab(emb_model);
    emb_dim   = llama_model_n_embd(emb_model);

    LOGI("Embedding model loaded. dim=%d", emb_dim);
}

void unload_embedding_model()
{
    if (emb_ctx)   { llama_free(emb_ctx);          emb_ctx   = nullptr; }
    if (emb_model) { llama_model_free(emb_model);  emb_model = nullptr; }
    emb_vocab = nullptr;
    emb_dim   = 0;
}

int get_embedding_dim() { return emb_dim; }

std::vector<float> embed_text(const std::string& text)
{
    if (!emb_model || !emb_ctx || !emb_vocab) {
        LOGE("embed_text: model not loaded");
        return {};
    }

    // ✅ OPTIONAL: truncate for speed (E5 works best with short text)
    std::string input = text;
    if (input.size() > 300) {
        input = input.substr(0, 300);
    }

    // 2-pass tokenize
    int n = llama_tokenize(
            emb_vocab,
            input.c_str(), (int)input.size(),
            nullptr, 0,
            true, false
    );
    if (n < 0) n = -n;

    std::vector<llama_token> tokens(n);

    llama_tokenize(
            emb_vocab,
            input.c_str(), (int)input.size(),
            tokens.data(), n,
            true, false
    );

    llama_memory_clear(llama_get_memory(emb_ctx), true);

    llama_batch batch = llama_batch_get_one(tokens.data(), n);
    if (llama_decode(emb_ctx, batch) != 0) {
        LOGE("embed_text: decode failed");
        return {};
    }

    const float* raw = llama_get_embeddings(emb_ctx);
    if (!raw) {
        LOGE("embed_text: null embeddings");
        return {};
    }

    std::vector<float> emb(raw, raw + emb_dim);

    float norm = 0.0f;
    for (float v : emb) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 1e-6f) {
        for (float& v : emb) v /= norm;
    }

    return emb;
}