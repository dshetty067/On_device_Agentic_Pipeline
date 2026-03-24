#include "llm_engine.h"
#include "llama.h"

#include <vector>
#include <string>
#include <thread>
#include <algorithm>
#include <fstream>
#include <android/log.h>

#define LOG_TAG "LLMEngine"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// ── Globals ────────────────────────────────────────────────────────────────
static llama_model*       g_model = nullptr;
static llama_context*     g_ctx   = nullptr;
static const llama_vocab* g_vocab = nullptr;

// ══════════════════════════════════════════════════════════════════════════════
// CONFIG
// ══════════════════════════════════════════════════════════════════════════════

static constexpr int CTX_SIZE = 2048;
static constexpr int BATCH_SIZE  = 768;
static constexpr int UBATCH_SIZE = 768;
// Max tokens to generate. For short RAG answers, 48 is plenty.
// Every token not generated = time saved.
static constexpr int MAX_NEW_TOKENS = 256;

// ══════════════════════════════════════════════════════════════════════════════
// P-CORE DETECTION
// Reads /sys/devices/system/cpu/cpuN/cpufreq/cpuinfo_max_freq to identify
// big cores by frequency. Falls back to hw_concurrency/2 if sysfs unavailable.
// ══════════════════════════════════════════════════════════════════════════════

static int detect_p_cores()
{
    // P-cores on modern Android SoCs run at >= 1.8 GHz max freq.
    // E-cores are typically <= 1.5 GHz.
    constexpr unsigned long P_CORE_MIN_KHZ = 1800000UL;

    int p_count = 0;
    for (int i = 0; i < 16; ++i) {
        std::string path = "/sys/devices/system/cpu/cpu"
                           + std::to_string(i)
                           + "/cpufreq/cpuinfo_max_freq";
        std::ifstream f(path);
        if (!f.is_open()) break;

        unsigned long freq = 0;
        f >> freq;
        if (freq >= P_CORE_MIN_KHZ) {
            ++p_count;
            LOGI("CPU%d = P-core @ %lu kHz", i, freq);
        } else {
            LOGI("CPU%d = E-core @ %lu kHz", i, freq);
        }
    }

    if (p_count <= 0) {
        // sysfs not readable — use conservative heuristic
        int hw = (int)std::thread::hardware_concurrency();
        if (hw >= 8) return 4;
        if (hw >= 6) return 3;
        return 2;
    }

    // Cap at 4: beyond 4 threads on mobile, memory bandwidth is the bottleneck.
    // More threads = more bus contention = slower decode, not faster.
    int result = std::min(p_count, 4);
    LOGI("Thread count: %d (detected %d P-cores, capped at 4)", result, p_count);
    return result;
}

static std::string trim(const std::string& s)
{
    size_t a = s.find_first_not_of(" \t\n\r");
    size_t b = s.find_last_not_of(" \t\n\r");
    return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
}

// ── load_model ─────────────────────────────────────────────────────────────

void load_model(const std::string& path)
{
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;     // CPU-only. For GPU: set to 99 and enable
    // GGML_VULKAN=ON in CMakeLists.txt → 3-5x faster.
    mparams.use_mmap     = true;  // Map weights from disk — avoids copying ~400MB into RAM.
    mparams.use_mlock    = false; // Do NOT lock pages — Android OOM killer will kill the app.

    g_model = llama_model_load_from_file(path.c_str(), mparams);
    if (!g_model) {
        LOGE("Failed to load model: %s", path.c_str());
        return;
    }
    g_vocab = llama_model_get_vocab(g_model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = CTX_SIZE;
    cparams.n_batch         = BATCH_SIZE;
    cparams.n_ubatch        = UBATCH_SIZE; // KEY FIX: was 16, now 512 = 1 prefill call

    int threads             = detect_p_cores();
    cparams.n_threads       = threads;
    cparams.n_threads_batch = threads;

    // Helps prefill ~10-15% on CPU. No effect on decode. Safe to keep.
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    // F16 KV cache (default). If RAM is very tight, try GGML_TYPE_Q8_0 here
    // to halve KV memory at a small quality cost.
    cparams.type_k = GGML_TYPE_F16;
    cparams.type_v = GGML_TYPE_F16;

    g_ctx = llama_init_from_model(g_model, cparams);
    if (!g_ctx) {
        LOGE("Failed to create llama context");
        return;
    }

    LOGI("Model loaded OK | ctx=%d | threads=%d | ubatch=%d", CTX_SIZE, threads, UBATCH_SIZE);
}

// ── Getters ────────────────────────────────────────────────────────────────
const llama_vocab* get_vocab() { return g_vocab; }
llama_context*     get_ctx()   { return g_ctx;   }

// ── generate ───────────────────────────────────────────────────────────────

std::string generate(const std::string& prompt,
                     std::function<void(const std::string&)> on_token)
{
    if (!g_model || !g_ctx || !g_vocab) return "[Error: model not loaded]";

    // Clear KV cache — required before each new turn.
    llama_memory_clear(llama_get_memory(g_ctx), true);

    // ── Tokenize ───────────────────────────────────────────────────────────
    int n = llama_tokenize(g_vocab, prompt.c_str(), (int)prompt.size(),
                           nullptr, 0, /*add_bos=*/true, /*special=*/true);
    if (n < 0) n = -n;

    std::vector<llama_token> tokens(n);
    llama_tokenize(g_vocab, prompt.c_str(), (int)prompt.size(),
                   tokens.data(), n, true, true);

    if (tokens.empty()) return "[Error: tokenization failed]";
    LOGI("Prompt: %d tokens", (int)tokens.size());

    // ── Truncate if needed ─────────────────────────────────────────────────
    const int n_ctx      = (int)llama_n_ctx(g_ctx);
    const int max_prompt = n_ctx - MAX_NEW_TOKENS;
    int n_tokens         = (int)tokens.size();

    if (n_tokens > max_prompt) {
        std::vector<llama_token> trunc;
        trunc.reserve(max_prompt);
        trunc.push_back(tokens[0]); // always keep BOS
        int tail = max_prompt - 1;
        trunc.insert(trunc.end(), tokens.end() - tail, tokens.end());
        tokens   = std::move(trunc);
        n_tokens = (int)tokens.size();
        LOGI("Prompt truncated to %d tokens", n_tokens);
    }

    // ── Prefill ────────────────────────────────────────────────────────────
    // Single call — UBATCH == BATCH == CTX_SIZE guarantees this.
    if (llama_decode(g_ctx, llama_batch_get_one(tokens.data(), n_tokens)) != 0) {
        return "[Error: prefill failed]";
    }

    // ── Sampler ────────────────────────────────────────────────────────────
    // USE_GREEDY_SAMPLER is set via add_compile_definitions() in CMakeLists.txt.
    // Greedy = pure argmax, zero math overhead per token.
    // Best choice for Qwen 0.5B RAG: model too small to benefit from sampling noise.
    auto           sparams = llama_sampler_chain_default_params();
    llama_sampler* smpl    = llama_sampler_chain_init(sparams);

#ifdef USE_GREEDY_SAMPLER
    // FASTEST + best quality for this model/task
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
#else
    // Minimal stochastic: temperature only (no top_k/top_p = 2 fewer ops/token)
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
#endif

    // ── Decode loop ────────────────────────────────────────────────────────
    std::string output;
    output.reserve(256);
    llama_token cur_token;

    for (int i = 0; i < MAX_NEW_TOKENS; ++i) {

        cur_token = llama_sampler_sample(smpl, g_ctx, -1);

        if (llama_vocab_is_eog(g_vocab, cur_token)) break;

        char buf[256];
        int piece_len = llama_token_to_piece(g_vocab, cur_token,
                                             buf, sizeof(buf), 0, false);
        if (piece_len <= 0) break;

        output.append(buf, piece_len);
        if (on_token) on_token(std::string(buf, piece_len));

        // Check only the tail — avoids scanning entire output string each token
        if (output.size() >= 10 &&
            output.compare(output.size() - 10, 10, "<|im_end|>") == 0) break;

        if (llama_decode(g_ctx, llama_batch_get_one(&cur_token, 1)) != 0) break;
    }

    llama_sampler_free(smpl);

    // Strip ChatML marker if it landed mid-output
    auto pos = output.find("<|im_end|>");
    if (pos != std::string::npos) output.erase(pos);

    std::string result = trim(output);
    LOGI("Output: %zu chars", result.size());
    return result.empty() ? "[No output]" : result;
}