#include "agent_bt.h"
#include "llm_engine.h"
#include "pdf_processor.h"
#include "embedding_engine.h"
#include "vector_store.h"

#include <behaviortree_cpp/bt_factory.h>
#include <android/log.h>
#include <sstream>
#include <fstream>

#define LOG_TAG "BT_AGENT"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

using namespace BT;

static AgentState state;

// ── Node 1: Load PDF, chunk it ─────────────────────────────────────────────
class PDFLoaderNode : public SyncActionNode {
public:
    PDFLoaderNode(const std::string& name, const NodeConfig& config,
                  std::function<void(const std::string&)> cb)
            : SyncActionNode(name, config), statusCb(cb) {}

    static BT::PortsList providedPorts() { return {}; }

    NodeStatus tick() override {
        if (statusCb) statusCb("🟡 Node 1/5: Loading PDF...");
        LOGI("PDFLoaderNode: loading %s", state.pdf_path.c_str());

        auto chunks = load_and_chunk_pdf(state.pdf_path);
        if (chunks.empty()) {
            if (statusCb) statusCb("❌ Node 1/5: PDF extraction failed");
            return NodeStatus::FAILURE;
        }

        std::ostringstream oss;
        oss << "✅ Node 1/5: PDF loaded — "
            << chunks.size() << " chunks extracted";
        LOGI("%s", oss.str().c_str());
        if (statusCb) statusCb(oss.str());

        static std::vector<std::string> s_chunks;
        s_chunks = chunks;

        for (int i = 0; i < std::min((int)chunks.size(), 3); i++) {
            LOGI("Chunk[%d]: %.80s...", i, chunks[i].c_str());
            if (statusCb) {
                statusCb("📄 Chunk " + std::to_string(i+1) + ": " +
                         chunks[i].substr(0, 80) + "...");
            }
        }

        return NodeStatus::SUCCESS;
    }
private:
    std::function<void(const std::string&)> statusCb;
};

// ── Node 2: Embed chunks + build vector store ──────────────────────────────
class EmbedNode : public SyncActionNode {
public:
    EmbedNode(const std::string& name, const NodeConfig& config,
              std::function<void(const std::string&)> cb)
            : SyncActionNode(name, config), statusCb(cb) {}

    static BT::PortsList providedPorts() { return {}; }

    NodeStatus tick() override {
        if (statusCb) statusCb("🟡 Node 2/5: Checking vector store...");
        LOGI("EmbedNode: starting");

        std::string bin = state.store_dir + "/vector_store.bin";
        std::ifstream check(bin);
        if (check.good()) {
            check.close();
            if (statusCb) statusCb("⚡ Node 2/5: Loading saved vector store...");
            if (vector_store_load(state.store_dir)) {
                if (statusCb) statusCb("✅ Node 2/5: Vector store loaded from disk");
                return NodeStatus::SUCCESS;
            }
        }

        // Use chunk_size=600, overlap=100 — matches pdf_processor defaults
        auto chunks = load_and_chunk_pdf(state.pdf_path, 400, 80);
        if (chunks.empty()) {
            if (statusCb) statusCb("❌ Node 2/5: No chunks to embed");
            return NodeStatus::FAILURE;
        }

        if (statusCb) statusCb("🟡 Node 2/5: Embedding " +
                               std::to_string(chunks.size()) + " chunks...");

        std::vector<std::vector<float>> embeddings;
        embeddings.reserve(chunks.size());

        // Track which chunks actually got embedded (some may be skipped)
        std::vector<std::string> embedded_chunks;
        embedded_chunks.reserve(chunks.size());

        for (int i = 0; i < (int)chunks.size(); i++) {
            // ✅ E5 requires "passage: " prefix for document chunks
            std::string input = "passage: " + chunks[i];

            auto emb = embed_text(input);
            if (emb.empty()) {
                LOGI("EmbedNode: skipping empty embedding at %d", i);
                continue;
            }

            embeddings.push_back(emb);
            embedded_chunks.push_back(chunks[i]);  // store original (no prefix)

            if (i % 5 == 0 || i == (int)chunks.size()-1) {
                std::string msg = "🔢 Embedded " + std::to_string(i+1) +
                                  "/" + std::to_string(chunks.size()) + " chunks";
                if (statusCb) statusCb(msg);
                LOGI("%s", msg.c_str());
            }
        }

        // ✅ Pass embedded_chunks (parallel to embeddings) to build
        vector_store_build(embeddings, embedded_chunks);
        vector_store_save(state.store_dir);

        if (statusCb) statusCb("✅ Node 2/5: Embeddings saved to storage");
        return NodeStatus::SUCCESS;
    }

private:
    std::function<void(const std::string&)> statusCb;
};

// ── Node 3: Log user prompt ────────────────────────────────────────────────
class PromptNode : public SyncActionNode {
public:
    PromptNode(const std::string& name, const NodeConfig& config,
               std::function<void(const std::string&)> cb)
            : SyncActionNode(name, config), statusCb(cb) {}

    static BT::PortsList providedPorts() { return {}; }

    NodeStatus tick() override {
        if (statusCb) statusCb("🟡 Node 3/5: Processing prompt...");
        LOGI("PromptNode: query = %s", state.query.c_str());
        if (statusCb) statusCb("💬 Query: " + state.query);
        return NodeStatus::SUCCESS;
    }
private:
    std::function<void(const std::string&)> statusCb;
};

// ── Node 4: Vector search ──────────────────────────────────────────────────
class VectorSearchNode : public SyncActionNode {
public:
    VectorSearchNode(const std::string& name, const NodeConfig& config,
                     std::function<void(const std::string&)> cb)
            : SyncActionNode(name, config), statusCb(cb) {}

    static BT::PortsList providedPorts() { return {}; }

    NodeStatus tick() override {
        if (statusCb) statusCb("🟡 Node 4/5: Searching vector DB...");
        LOGI("VectorSearchNode: embedding query");

        // ✅ E5 requires "query: " prefix for search queries
        std::string query_input = "query: " + state.query;
        auto query_emb = embed_text(query_input);
        if (query_emb.empty()) {
            if (statusCb) statusCb("❌ Node 4/5: Query embedding failed");
            return NodeStatus::FAILURE;
        }

        // Retrieve top-5 for better coverage, then feed all to LLM
        auto results = vector_store_search(query_emb, 5);
        if (results.empty()) {
            if (statusCb) statusCb("⚠️ Node 4/5: No results found");
            state.retrieved_context = "";
            return NodeStatus::SUCCESS;
        }

        std::ostringstream ctx;
        for (int i = 0; i < (int)results.size(); i++) {
            std::string msg = "📌 Retrieved [" + std::to_string(i+1) +
                              "] score=" +
                              std::to_string(results[i].score).substr(0,5) +
                              " → " + results[i].text.substr(0, 120) + "...";
            if (statusCb) statusCb(msg);
            LOGI("%s", msg.c_str());
            ctx << "[Context " << (i+1) << "]: " << results[i].text << "\n\n";
        }

        state.retrieved_context = ctx.str();
        if (statusCb) statusCb("✅ Node 4/5: Context retrieved");
        return NodeStatus::SUCCESS;
    }
private:
    std::function<void(const std::string&)> statusCb;
};

// ── Node 5: LLM generation with RAG context ────────────────────────────────
class LLMNode : public SyncActionNode {
public:
    LLMNode(const std::string& name, const NodeConfig& config,
            std::function<void(const std::string&)> tokenCb,
            std::function<void(const std::string&)> statusCb)
            : SyncActionNode(name, config),
              tokenCallback(tokenCb), statusCallback(statusCb) {}

    static BT::PortsList providedPorts() { return {}; }

    NodeStatus tick() override
    {
        const llama_vocab* vocab = get_vocab();
        llama_context*     ctx   = get_ctx();

        if (!vocab || !ctx) {
            if (statusCallback) statusCallback("❌ Model not loaded");
            return NodeStatus::FAILURE;
        }

        if (statusCallback) statusCallback("🟡 LLM generating...");

        // ✅ Increased context chars — CTX_SIZE is now 768, so we can afford more
        // ~500 chars ≈ 125 tokens, leaving plenty of room for 256 generation tokens
        static constexpr size_t MAX_CONTEXT_CHARS  = 2000;
        static constexpr size_t MAX_QUESTION_CHARS = 150;

        std::string context  = state.retrieved_context;
        std::string question = state.query;

        // Trim context at a sentence boundary if possible
        if (context.size() > MAX_CONTEXT_CHARS) {
            context = context.substr(0, MAX_CONTEXT_CHARS);
            // Try to cut at last sentence end
            size_t last_dot = context.find_last_of(".!?");
            if (last_dot != std::string::npos && last_dot > MAX_CONTEXT_CHARS / 2)
                context = context.substr(0, last_dot + 1);
        }
        if (question.size() > MAX_QUESTION_CHARS)
            question = question.substr(0, MAX_QUESTION_CHARS);

        // ✅ Better system prompt: instructs the model to answer fully
        const std::string prompt =
                "<|im_start|>system\n"
                "You are a strict question-answering assistant.\n"
                "Answer ONLY using the provided context.\n"
                "Do NOT use outside knowledge.\n"
                "If the answer is not in the context, say: \"Not found in document.\"\n"
                "<|im_end|>\n"

                "<|im_start|>user\n"
                "CONTEXT:\n"
                "---------------------\n" +
                context +
                "---------------------\n\n"

                "QUESTION:\n" + question + "\n"
                                           "<|im_end|>\n"

                                           "<|im_start|>assistant\n";

        const std::string result = generate(prompt, tokenCallback);

        if (result.empty() || result[0] == '[') {
            if (statusCallback) statusCallback("❌ LLM failed — " + result);
            return NodeStatus::FAILURE;
        }

        state.response = result;

        std::string preview = result.substr(0, std::min((size_t)80, result.size()));
        if (statusCallback) statusCallback("✅ Done — " + preview);
        return NodeStatus::SUCCESS;
    }

private:
    std::function<void(const std::string&)> tokenCallback;
    std::function<void(const std::string&)> statusCallback;
};

// ── run_agent ──────────────────────────────────────────────────────────────
void run_agent(
        const std::string& pdf_path,
        const std::string& store_dir,
        const std::string& prompt,
        std::function<void(const std::string&)> on_token,
        std::function<void(const std::string&)> on_status)
{
    state.pdf_path  = pdf_path;
    state.store_dir = store_dir;
    state.query     = prompt;
    state.retrieved_context = "";
    state.response  = "";

    BehaviorTreeFactory factory;

    factory.registerBuilder<PDFLoaderNode>("PDFLoader",
                                           [on_status](const std::string& name, const NodeConfig& cfg) {
                                               return std::make_unique<PDFLoaderNode>(name, cfg, on_status);
                                           });

    factory.registerBuilder<EmbedNode>("Embed",
                                       [on_status](const std::string& name, const NodeConfig& cfg) {
                                           return std::make_unique<EmbedNode>(name, cfg, on_status);
                                       });

    factory.registerBuilder<PromptNode>("Prompt",
                                        [on_status](const std::string& name, const NodeConfig& cfg) {
                                            return std::make_unique<PromptNode>(name, cfg, on_status);
                                        });

    factory.registerBuilder<VectorSearchNode>("VectorSearch",
                                              [on_status](const std::string& name, const NodeConfig& cfg) {
                                                  return std::make_unique<VectorSearchNode>(name, cfg, on_status);
                                              });

    factory.registerBuilder<LLMNode>("LLM",
                                     [on_token, on_status](const std::string& name, const NodeConfig& cfg) {
                                         return std::make_unique<LLMNode>(name, cfg, on_token, on_status);
                                     });

    static const char* xml = R"(
<root BTCPP_format="4">
  <BehaviorTree ID="RAGTree">
    <Sequence>
      <PDFLoader/>
      <Embed/>
      <Prompt/>
      <VectorSearch/>
      <LLM/>
    </Sequence>
  </BehaviorTree>
</root>
)";

    auto tree = factory.createTreeFromText(xml);
    tree.tickWhileRunning();
}