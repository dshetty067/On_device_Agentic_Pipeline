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

static llama_model*       g_model = nullptr;
static llama_context*     g_ctx   = nullptr;
static const llama_vocab* g_vocab = nullptr;

static constexpr int CTX_SIZE  = 2048;
static constexpr int BATCH_SIZE = 512;

static int detect_p_cores() {
    constexpr unsigned long P_CORE_MIN_KHZ = 1800000UL;
    int p = 0;
    for (int i = 0; i < 16; ++i) {
        std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(i) + "/cpufreq/cpuinfo_max_freq";
        std::ifstream f(path);
        if (!f.is_open()) break;
        unsigned long freq = 0; f >> freq;
        if (freq >= P_CORE_MIN_KHZ) ++p;
    }
    if (p <= 0) {
        int hw = (int)std::thread::hardware_concurrency();
        return hw >= 8 ? 4 : (hw >= 6 ? 3 : 2);
    }
    return std::min(p, 4);
}

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\n\r");
    size_t b = s.find_last_not_of(" \t\n\r");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

// ── Internal generate (reused by all 3 public functions) ───────────────────
static std::string _generate(const std::string& prompt,
                             int max_tokens,
                             std::function<void(const std::string&)> on_token = nullptr)
{
    if (!g_model || !g_ctx || !g_vocab) return "[model not loaded]";

    llama_memory_clear(llama_get_memory(g_ctx), true);

    int n = llama_tokenize(g_vocab, prompt.c_str(), (int)prompt.size(), nullptr, 0, true, true);
    if (n < 0) n = -n;
    std::vector<llama_token> tokens(n);
    llama_tokenize(g_vocab, prompt.c_str(), (int)prompt.size(), tokens.data(), n, true, true);
    if (tokens.empty()) return "[tokenize failed]";

    // Truncate if needed
    // Truncate if needed — MUST leave room for max_tokens output
    int max_prompt = CTX_SIZE - max_tokens - 4;
    if ((int)tokens.size() > max_prompt) {
        LOGI("_generate: prompt truncated %d → %d tokens (ctx=%d max_out=%d)",
             (int)tokens.size(), max_prompt, CTX_SIZE, max_tokens);

        std::vector<llama_token> t;
        t.push_back(tokens[0]);
        t.insert(t.end(), tokens.end() - (max_prompt - 1), tokens.end());
        tokens = std::move(t);
    }

// Final hard check — ggml_abort fires if this is violated
    LOGI("_generate: %d prompt tokens, %d max output, ctx=%d",
         (int)tokens.size(), max_tokens, CTX_SIZE);
    if ((int)tokens.size() + max_tokens > CTX_SIZE) {
        LOGE("_generate: still over budget after truncation — capping output");
        max_tokens = CTX_SIZE - (int)tokens.size() - 2;
        if (max_tokens <= 0) return "[prompt too long]";
    }
    if (llama_decode(g_ctx, llama_batch_get_one(tokens.data(), (int)tokens.size())) != 0)
        return "[prefill failed]";

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sparams);
    // Greedy for intent/tool calls (deterministic), temperature for answers
    if (on_token) {
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    }

    std::string out;
    out.reserve(256);
    for (int i = 0; i < max_tokens; ++i) {
        llama_token tok = llama_sampler_sample(smpl, g_ctx, -1);
        if (llama_vocab_is_eog(g_vocab, tok)) break;
        char buf[256];
        int len = llama_token_to_piece(g_vocab, tok, buf, sizeof(buf), 0, false);
        if (len <= 0) break;
        out.append(buf, len);
        if (on_token) on_token(std::string(buf, len));
        if (out.size() >= 10 && out.compare(out.size()-10, 10, "<|im_end|>") == 0) break;
        if (llama_decode(g_ctx, llama_batch_get_one(&tok, 1)) != 0) break;
    }
    llama_sampler_free(smpl);

    auto p = out.find("<|im_end|>");
    if (p != std::string::npos) out.erase(p);
    return trim(out);
}

// ── load_model ─────────────────────────────────────────────────────────────
void load_model(const std::string& path) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0; mp.use_mmap = true; mp.use_mlock = false;
    g_model = llama_model_load_from_file(path.c_str(), mp);
    if (!g_model) { LOGE("load failed: %s", path.c_str()); return; }
    g_vocab = llama_model_get_vocab(g_model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = CTX_SIZE; cp.n_batch = BATCH_SIZE; cp.n_ubatch = BATCH_SIZE;
    int th = detect_p_cores();
    cp.n_threads = std::min(th, 4); cp.n_threads_batch = th;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = GGML_TYPE_F16; cp.type_v = GGML_TYPE_F16;
    g_ctx = llama_init_from_model(g_model, cp);
    LOGI("Model loaded | ctx=%d threads=%d", CTX_SIZE, th);
}


std::string classify_intent(const std::string& query) {
    std::string prompt =

            "If a tool is needed: {\" \"<|im_start|>system\\n\"\n"
            "            \"You are an intent classifier. Respond ONLY with valid JSON, no extra text.For the given user qury which tool is best suitable that you should identify\\n\"\n"
            "            \"Available tools:\\n\"\n"
            "            \"- web_search: get current weather of any place and current news and affairs\\n\"\n"
            "            \"- rag: answer questions related to stock market performance in 2024\\n\"\n"
            "            \"- book_flight: book a flight ticket\\n\"action\":\"use_tool\",\"tool\":\"<name>\"}\n"
            "If no tool needed: {\"action\":\"general_answer\"}\n"
            "<|im_end|>\n"
            "<|im_start|>user\n" + query + "\n<|im_end|>\n"
                                           "<|im_start|>assistant\n";
    return _generate(prompt, 64);
}


std::string general_answer(const std::string &query,
                           std::function<void(const std::string &)> on_token) {
    std::string prompt =
            "<|im_start|>system\n"
            "You are a helpful assistant. Answer clearly and concisely.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n" + query + "\n<|im_end|>\n"
                                           "<|im_start|>assistant\n";
    return _generate(prompt, 256, on_token);
}


std::string refined_answer(const std::string& query,
                           const std::string& tool_name,
                           const std::string& tool_result,
                           std::function<void(const std::string&)> on_token) {
    std::string prompt =
            "<|im_start|>system\n"
            "You are a helpful assistant. Use the tool result to answer the user.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            "Question: " + query + "\n"
                                   "Tool (" + tool_name + ") result: " + tool_result + "\n"
                                                                                       "<|im_end|>\n"
                                                                                       "<|im_start|>assistant\n";
    return _generate(prompt, 256, on_token);
}