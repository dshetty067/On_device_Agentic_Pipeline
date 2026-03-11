#include "agent_bt.h"
#include "llm_engine.h"
#include "pdf_processor.h"
#include "embedding_engine.h"
#include "vector_store.h"

#include <behaviortree_cpp/bt_factory.h>
#include <android/log.h>
#include <sstream>
#include <fstream>    // ← add this line

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

        // Log chunk summary
        std::ostringstream oss;
        oss << "✅ Node 1/5: PDF loaded — "
            << chunks.size() << " chunks extracted";
        LOGI("%s", oss.str().c_str());
        if (statusCb) statusCb(oss.str());

        // Store chunks globally for next node
        // We reuse vector_store's chunk list via rebuild in EmbedNode
        // Pass chunks via a static local (simple approach)
        static std::vector<std::string> s_chunks;
        s_chunks = chunks;

        // Log first 3 chunks
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
        if (statusCb) statusCb("🟡 Node 2/5: Loading embedding model...");
        LOGI("EmbedNode: starting");

        // Check if vector store already saved for this PDF
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

        // Re-chunk (chunks were extracted in PDFLoaderNode)
        auto chunks = load_and_chunk_pdf(state.pdf_path);
        if (chunks.empty()) {
            if (statusCb) statusCb("❌ Node 2/5: No chunks to embed");
            return NodeStatus::FAILURE;
        }

        if (statusCb) statusCb("🟡 Node 2/5: Embedding " +
                               std::to_string(chunks.size()) + " chunks...");

        std::vector<std::vector<float>> embeddings;
        embeddings.reserve(chunks.size());

        for (int i = 0; i < (int)chunks.size(); i++) {
            auto emb = embed_text(chunks[i]);
            if (emb.empty()) {
                LOGI("EmbedNode: skipping empty embedding at %d", i);
                continue;
            }
            embeddings.push_back(emb);

            if (i % 5 == 0 || i == (int)chunks.size()-1) {
                std::string msg = "🔢 Embedded " + std::to_string(i+1) +
                                  "/" + std::to_string(chunks.size()) + " chunks";
                if (statusCb) statusCb(msg);
                LOGI("%s", msg.c_str());
            }
        }

        vector_store_build(embeddings, chunks);
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

        auto query_emb = embed_text(state.query);
        if (query_emb.empty()) {
            if (statusCb) statusCb("❌ Node 4/5: Query embedding failed");
            return NodeStatus::FAILURE;
        }

        auto results = vector_store_search(query_emb, 3);
        if (results.empty()) {
            if (statusCb) statusCb("⚠️ Node 4/5: No results found");
            state.retrieved_context = "";
            return NodeStatus::SUCCESS;
        }

        // Build context string + show on screen
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

    NodeStatus tick() override {
        if (statusCallback) statusCallback("🟡 Node 5/5: LLM generating...");
        LOGI("LLMNode: building RAG prompt");

        // Build RAG prompt
        std::string rag_prompt =
                "You are a helpful assistant. Use the following context to answer "
                "the user's question.\n\n"
                "Context:\n" + state.retrieved_context +
                "\nQuestion: " + state.query +
                "\nAnswer:";

        LOGI("RAG prompt length: %zu", rag_prompt.size());

        state.response = generate(rag_prompt, tokenCallback);

        if (statusCallback) statusCallback("✅ Node 5/5: Done");
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