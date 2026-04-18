#include "embedding_engine.h"
#include <android/log.h>
#include <cmath>
#include <cstring>
#include <thread>
#include <numeric>

#define LOG_TAG "EmbeddingEngine"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static llama_model*       emb_model = nullptr;
static llama_context*     emb_ctx   = nullptr;
static const llama_vocab* emb_vocab = nullptr;
static int                emb_dim   = 0;

// ── load ───────────────────────────────────────────────────────────────────
void load_embedding_model(const std::string& path)
{
    LOGI("load_embedding_model: %s", path.c_str());

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.use_mmap     = true;

    emb_model = llama_model_load_from_file(path.c_str(), mparams);
    if (!emb_model) { LOGE("Failed to load embedding model"); return; }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx        = 512;          // E5-small max is 512
    cparams.n_batch      = 512;
    cparams.n_threads    = std::max(2, (int)std::thread::hardware_concurrency() / 2);
    cparams.embeddings   = true;

    // *** KEY FIX: use MEAN pooling — this is what multilingual-e5 expects ***
    cparams.pooling_type = LLAMA_POOLING_TYPE_MEAN;

    emb_ctx = llama_init_from_model(emb_model, cparams);
    if (!emb_ctx) { LOGE("Failed to create embedding context"); return; }

    emb_vocab = llama_model_get_vocab(emb_model);
    emb_dim   = llama_model_n_embd(emb_model);

    LOGI("Embedding model loaded. dim=%d  pooling=MEAN", emb_dim);
}

void unload_embedding_model()
{
    if (emb_ctx)   { llama_free(emb_ctx);         emb_ctx   = nullptr; }
    if (emb_model) { llama_model_free(emb_model); emb_model = nullptr; }
    emb_vocab = nullptr;
    emb_dim   = 0;
}

int get_embedding_dim() { return emb_dim; }

// ── embed_text ─────────────────────────────────────────────────────────────
//
// CRITICAL FIXES vs old version:
//  1. Truncate input to 500 chars to avoid exceeding n_ctx=512 after BPE.
//  2. Use llama_get_embeddings_seq(ctx, 0) instead of llama_get_embeddings —
//     with MEAN pooling the sequence embedding is at seq_id=0, not position 0.
//  3. L2-normalise the output so cosine sim == inner product (InnerProductSpace).
//  4. Log the first few values so you can confirm non-zero output.
//
std::vector<float> embed_text(const std::string& text)
{
    if (!emb_model || !emb_ctx || !emb_vocab) {
        LOGE("embed_text: model not loaded");
        return {};
    }

    // Truncate to stay inside context window (BPE can expand chars 3-4x)
    std::string input = text;
    if (input.size() > 500) {
        input = input.substr(0, 500);
        LOGI("embed_text: input truncated to 500 chars");
    }

    // 2-pass tokenise
    int n = llama_tokenize(emb_vocab, input.c_str(), (int)input.size(),
                           nullptr, 0, true, false);
    if (n < 0) n = -n;
    if (n == 0) { LOGE("embed_text: no tokens"); return {}; }

    std::vector<llama_token> tokens(n);
    llama_tokenize(emb_vocab, input.c_str(), (int)input.size(),
                   tokens.data(), n, true, false);

    // Cap at context limit
    int max_tok = (int)llama_n_ctx(emb_ctx) - 2;
    if (n > max_tok) {
        LOGI("embed_text: trimmed %d → %d tokens", n, max_tok);
        tokens.resize(max_tok);
        n = max_tok;
    }

    LOGI("embed_text: '%s...' → %d tokens", input.substr(0, 60).c_str(), n);

    llama_memory_clear(llama_get_memory(emb_ctx), true);

    // Build batch with seq_id=0 so MEAN pooling aggregates correctly
    llama_batch batch = llama_batch_init(n, 0, 1);
    batch.n_tokens = n;
    for (int i = 0; i < n; i++) {
        batch.token   [i] = tokens[i];
        batch.pos     [i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id  [i][0] = 0;
        batch.logits  [i] = (i == n - 1) ? 1 : 0; // only last needed for CLS; MEAN uses all
    }
    // For MEAN pooling llama.cpp needs ALL logits flagged
    for (int i = 0; i < n; i++) batch.logits[i] = 1;

    if (llama_decode(emb_ctx, batch) != 0) {
        LOGE("embed_text: decode failed");
        llama_batch_free(batch);
        return {};
    }

    // *** Use llama_get_embeddings_seq for MEAN-pooled vector ***
    const float* raw = llama_get_embeddings_seq(emb_ctx, 0);
    if (!raw) {
        // Fallback: try per-token embedding of last token
        LOGI("embed_text: llama_get_embeddings_seq returned null, trying per-token fallback");
        raw = llama_get_embeddings_ith(emb_ctx, n - 1);
    }
    if (!raw) {
        LOGE("embed_text: all embedding retrieval methods failed");
        llama_batch_free(batch);
        return {};
    }

    std::vector<float> emb(raw, raw + emb_dim);
    llama_batch_free(batch);

    // L2 normalise → cosine sim = inner product
    float norm = 0.0f;
    for (float v : emb) norm += v * v;
    norm = std::sqrt(norm);
    if (norm < 1e-6f) {
        LOGE("embed_text: zero-norm embedding — model may not support embeddings");
        return {};
    }
    for (float& v : emb) v /= norm;

    // Sanity-check: log first 4 values
    LOGI("embed_text: OK  dim=%d  first4=[%.4f, %.4f, %.4f, %.4f]  norm_before=%.4f",
         emb_dim, emb[0], emb[1], emb[2], emb[3], norm);

    return emb;
}

std::vector<float> embed_query(const std::string& q) {
    return embed_text("query: " + q);
}

std::vector<float> embed_passage(const std::string& p) {
    return embed_text("passage: " + p);
}