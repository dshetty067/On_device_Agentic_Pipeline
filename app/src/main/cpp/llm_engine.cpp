#include "llm_engine.h"
#include "llama.h"

#include <vector>
#include <string>
#include <android/log.h>

#define LOG_TAG "LLMEngine"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static llama_model*   model = nullptr;
static llama_context* ctx   = nullptr;
static const llama_vocab* vocab = nullptr;

void load_model(const std::string& path)
{
    LOGI("load_model: starting");

    llama_model_params mparams = llama_model_default_params();
    model = llama_model_load_from_file(path.c_str(), mparams);

    if (!model) {
        LOGE("load_model: FAILED to load model from %s", path.c_str());
        return;
    }
    LOGI("load_model: model loaded OK");

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = 512;   // keep small on mobile
    cparams.n_batch    = 512;
    cparams.n_threads  = 4;

    ctx = llama_init_from_model(model, cparams);

    if (!ctx) {
        LOGE("load_model: FAILED to create context");
        return;
    }

    vocab = llama_model_get_vocab(model);
    LOGI("load_model: context ready, vocab ready");
}

std::string generate(const std::string& prompt,std::function<void(const std::string&)> on_token)
{
    if (!model || !ctx || !vocab) {
        LOGE("generate: model/ctx/vocab not initialized");
        return "[Error: model not loaded]";
    }

    LOGI("generate: starting, prompt length=%zu", prompt.size());

    // ── Always clear KV cache before a new generation ──
    llama_memory_clear(llama_get_memory(ctx), true);
    LOGI("generate: KV cache cleared");

    // ── Tokenize ───────────────────────────────────────
    std::vector<llama_token> tokens(prompt.size() + 64);
    int n_tokens = llama_tokenize(
            vocab,
            prompt.c_str(),
            (int)prompt.size(),
            tokens.data(),
            (int)tokens.size(),
            true,   // add BOS
            true    // special tokens
    );

    if (n_tokens < 0) {
        LOGE("generate: tokenize failed, n_tokens=%d", n_tokens);
        return "[Error: tokenization failed]";
    }
    tokens.resize(n_tokens);
    LOGI("generate: tokenized OK, n_tokens=%d", n_tokens);

    // ── Prefill / prompt eval ──────────────────────────
    llama_batch batch = llama_batch_get_one(tokens.data(), (int)tokens.size());
    int ret = llama_decode(ctx, batch);
    if (ret != 0) {
        LOGE("generate: llama_decode (prefill) failed, ret=%d", ret);
        return "[Error: decode failed]";
    }
    LOGI("generate: prefill done");

    // ── Sampler setup ──────────────────────────────────
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // ── Generation loop ────────────────────────────────
    std::string output;
    const int max_tokens = 100; // reduce from 200 to 100

    LOGI("generate: starting token loop (max=%d)", max_tokens);

    for (int i = 0; i < max_tokens; i++)
    {
        llama_token token = llama_sampler_sample(smpl, ctx, -1);

        if (llama_vocab_is_eog(vocab, token)) {
            LOGI("generate: EOG at token %d", i);
            break;
        }

        char buf[256];
        int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
        if (n > 0) {
            std::string piece(buf, n);
            output += piece;

            // Stream each token back immediately
            if (on_token) {
                on_token(piece);
            }
        }

        llama_batch next = llama_batch_get_one(&token, 1);
        ret = llama_decode(ctx, next);
        if (ret != 0) {
            LOGE("generate: llama_decode failed at token %d, ret=%d", i, ret);
            break;
        }

        if (i % 20 == 0)
            LOGI("generate: token %d, output so far: %zu chars", i, output.size());
    }

    llama_sampler_free(smpl);
    LOGI("generate: done, output length=%zu", output.size());
    return output.empty() ? "[No output generated]" : output;
}