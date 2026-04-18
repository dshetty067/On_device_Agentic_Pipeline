#include "agent_bt.h"
#include "blackboard.h"
#include "llm_engine.h"
#include "pdf_processor.h"
#include "embedding_engine.h"
#include "vector_store.h"

#include <behaviortree_cpp/bt_factory.h>
#include <android/log.h>
#include <sstream>
#include <algorithm>
#include <jni.h>
#include <fstream>

#define LOG_TAG "BT_AGENT"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

using namespace BT;

static JavaVM* g_jvm = nullptr;
static jobject g_activity = nullptr;

// ── Shared state ───────────────────────────────────────────────────────────
static AgentBlackboard g_bb;

static std::function<void(const std::string&)> g_on_token;
static std::function<void(const std::string&)> g_on_status;

// PDF / RAG paths (set once by MainActivity via JNI)
static std::string g_pdf_path;
static std::string g_index_dir;

// Raw embeddings kept in memory after first build so rerank works
// without hitting disk on every query.
static std::vector<std::vector<float>> g_stored_embeddings;

// ── Setters (called from JNI) ──────────────────────────────────────────────
void agent_set_pdf_path(const std::string& path) {
    g_pdf_path = path;
    LOGI("agent_set_pdf_path: %s", path.c_str());
}

void agent_set_index_dir(const std::string& dir) {
    g_index_dir = dir;
    LOGI("agent_set_index_dir: %s", dir.c_str());
}

void agent_set_jni(JavaVM* jvm, jobject activity) {
    g_jvm = jvm;
    g_activity = activity;
}

// ── Helpers ────────────────────────────────────────────────────────────────
static void logStatus(const std::string& msg) {
    LOGI("%s", msg.c_str());
    if (g_on_status) g_on_status(msg);
}

// Tiny JSON extractor – no external deps.
// Handles flat {"key":"value"} objects produced by classify_intent().
static std::string jsonGet(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    size_t kp = json.find(needle);
    if (kp == std::string::npos) return "";
    size_t colon = json.find(':', kp + needle.size());
    if (colon == std::string::npos) return "";
    size_t q1 = json.find('"', colon + 1);
    if (q1 == std::string::npos) return "";
    size_t q2 = json.find('"', q1 + 1);
    if (q2 == std::string::npos) return "";
    return json.substr(q1 + 1, q2 - q1 - 1);
}

// ════════════════════════════════════════════════════════════════════════════
// NODE 1 – IntentNode
// Calls classify_intent(), parses JSON, writes planned_action / selected_tool.
// ════════════════════════════════════════════════════════════════════════════
class IntentNode : public SyncActionNode {
public:
    IntentNode(const std::string& name, const NodeConfig& cfg)
            : SyncActionNode(name, cfg) {}
    static BT::PortsList providedPorts() { return {}; }

    NodeStatus tick() override {
        logStatus("🟡 [1/3] Classifying intent...");
        std::string raw = classify_intent(g_bb.user_query);
        LOGI("IntentNode raw: [%s]", raw.c_str());

        std::string action = jsonGet(raw, "action");
        std::string tool   = jsonGet(raw, "tool");

        // Handle both formats:
        // {"action":"use_tool","tool":"web_search"}  ← expected
        // {"action":"web_search"}                    ← what you actually get
        static const std::set<std::string> known_tools = {
                "web_search", "rag", "book_flight"
        };

        if (!action.empty() && known_tools.count(action)) {
            // action IS the tool name — normalize it
            g_bb.planned_action = "use_tool";
            g_bb.selected_tool  = action;
        } else {
            g_bb.planned_action = action.empty() ? "general_answer" : action;
            g_bb.selected_tool  = tool;
        }

        if (g_bb.planned_action.empty()) {
            g_bb.planned_action = "general_answer";
            logStatus("⚠️ [1/3] Intent parse failed — defaulting to general_answer");
        } else {
            logStatus("✅ [1/3] Intent: " + g_bb.planned_action +
                      (g_bb.selected_tool.empty() ? "" : " → " + g_bb.selected_tool));
        }
        return NodeStatus::SUCCESS;
    }
};

// ════════════════════════════════════════════════════════════════════════════
// CONDITION – ShouldUseTool
// ════════════════════════════════════════════════════════════════════════════
class ShouldUseToolNode : public ConditionNode {
public:
    ShouldUseToolNode(const std::string& name, const NodeConfig& cfg)
            : ConditionNode(name, cfg) {}
    static BT::PortsList providedPorts() { return {}; }

    NodeStatus tick() override {
        bool v = (g_bb.planned_action == "use_tool" && !g_bb.selected_tool.empty());
        logStatus(v ? "🔧 Path → tool" : "💬 Path → general answer");
        return v ? NodeStatus::SUCCESS : NodeStatus::FAILURE;
    }
};

// ════════════════════════════════════════════════════════════════════════════
// NODE 2 – ToolExecutorNode
// Runs the chosen tool; writes tool_result + tool_succeeded to blackboard.
// ════════════════════════════════════════════════════════════════════════════

class ToolExecutorNode : public SyncActionNode {
public:
    ToolExecutorNode(const std::string& name, const NodeConfig& cfg)
            : SyncActionNode(name, cfg) {}

    static BT::PortsList providedPorts() { return {}; }

    NodeStatus tick() override {
        const std::string& tool = g_bb.selected_tool;
        logStatus("🔧 [2/3] Executing tool: " + tool);

        if (tool == "web_search") {
            NodeStatus s = executeWebSearch();
            if (s != NodeStatus::SUCCESS) return s;

        } else if (tool == "rag") {
            NodeStatus s = executeRag();
            if (s != NodeStatus::SUCCESS) return s;

        } else if (tool == "book_flight") {
            g_bb.tool_result    = "Ticket booked successfully.";
            g_bb.tool_succeeded = true;

        } else {
            g_bb.tool_result    = "Unknown tool: " + tool;
            g_bb.tool_succeeded = false;
            return NodeStatus::FAILURE;
        }

        logStatus("✅ [2/3] Tool result ready");
        return NodeStatus::SUCCESS;
    }

    // 🔥 NEW JNI WEB SEARCH
    NodeStatus executeWebSearch() {
        logStatus("🟡 [2/3] Web search via Kotlin...");

        if (!g_jvm || !g_activity) {
            g_bb.tool_result    = "Web search is not available (JNI not initialized).";
            g_bb.tool_succeeded = false;
            return NodeStatus::FAILURE;
        }

        JNIEnv* env = nullptr;
        g_jvm->AttachCurrentThread(&env, nullptr);

        jclass cls = env->GetObjectClass(g_activity);
        jmethodID method = env->GetMethodID(
                cls, "fetchWebSearch", "(Ljava/lang/String;)Ljava/lang/String;");

        if (!method) {
            g_bb.tool_result    = "fetchWebSearch method not found.";
            g_bb.tool_succeeded = false;
            return NodeStatus::FAILURE;
        }

        jstring jquery  = env->NewStringUTF(g_bb.user_query.c_str());
        jstring jresult = (jstring)env->CallObjectMethod(g_activity, method, jquery);
        env->DeleteLocalRef(jquery);

        if (!jresult) {
            g_bb.tool_result    = "Web search returned null.";
            g_bb.tool_succeeded = false;
            return NodeStatus::FAILURE;
        }

        const char* res = env->GetStringUTFChars(jresult, nullptr);
        std::string result(res);
        env->ReleaseStringUTFChars(jresult, res);
        env->DeleteLocalRef(jresult);

        // ── NEW: treat "no results" as a soft failure ─────────────────────────
        static const std::vector<std::string> failure_markers = {
                "No search results found",
                "Search error:",
                "Web search error:"
        };
        for (const auto& marker : failure_markers) {
            if (result.find(marker) != std::string::npos) {
                g_bb.tool_result    = result; // keep the message
                g_bb.tool_succeeded = false;
                logStatus("⚠️ [2/3] Web search returned no useful data");
                return NodeStatus::FAILURE; // fall through to GeneralAnswer
            }
        }

        g_bb.tool_result    = result;
        g_bb.tool_succeeded = true;
        logStatus("✅ [2/3] Web result received (" +
                  std::to_string(result.size()) + " chars)");
        return NodeStatus::SUCCESS;
    }
private:
    // ── Full RAG pipeline ─────────────────────────────────────────────────
    NodeStatus executeRag() {

        // ── Guard: need a PDF path ────────────────────────────────────────
        if (g_pdf_path.empty()) {
            g_bb.tool_result    = "No document has been loaded. "
                                  "Please tap 📄 to upload a PDF first.";
            g_bb.tool_succeeded = false;
            logStatus("❌ [2/3] RAG: no PDF path set");
            return NodeStatus::FAILURE;
        }

        // ── Step 1: load existing index OR build a new one ────────────────
        if (!vector_store_is_ready()) {
            bool loaded = !g_index_dir.empty() && vector_store_load(g_index_dir);

            if (loaded && get_embedding_dim() > 0) {
                loadEmbeddingsFromDisk();
                // If embeddings failed to load or are zero-dim, rebuild
                if (g_stored_embeddings.empty()) {
                    LOGI("RAG: embeddings missing after load — rebuilding");
                    if (!buildIndex()) return NodeStatus::FAILURE;
                } else {
                    logStatus("✅ [2/3] RAG: index loaded from disk");
                }
            } else {
                // dim=0 means index was saved before embedding model was ready
                if (loaded) LOGI("RAG: stale index (dim=0) — rebuilding");
                if (!buildIndex()) return NodeStatus::FAILURE;
            }
        } else {
            if (g_stored_embeddings.empty() && !g_index_dir.empty())
                loadEmbeddingsFromDisk();
        }

        // ── Step 2: embed the query ───────────────────────────────────────
        logStatus("🟡 [2/3] RAG: embedding query...");
        auto q_emb = embed_query(g_bb.user_query);
        if (q_emb.empty()) {
            g_bb.tool_result    = "Failed to embed the query. "
                                  "Check that the embedding model is loaded.";
            g_bb.tool_succeeded = false;
            logStatus("❌ [2/3] RAG: query embedding failed");
            return NodeStatus::FAILURE;
        }

        // ── Step 3: ANN search (top 10) ───────────────────────────────────
        logStatus("🟡 [2/3] RAG: searching...");
        auto results = vector_store_search(q_emb, 5);

        if (results.empty()) {
            g_bb.tool_result    = "No relevant content found in the document.";
            g_bb.tool_succeeded = false;
            logStatus("❌ [2/3] RAG: no results returned");
            return NodeStatus::FAILURE;
        }

        // ── Step 4: rerank with exact cosine sim ──────────────────────────
        if (!g_stored_embeddings.empty()) {
            results = rerank_results(q_emb, results, g_stored_embeddings);
            logStatus("✅ [2/3] RAG: reranked " +
                      std::to_string(results.size()) + " candidates");
        }

        // ── Step 5: take top 3 and build context string ───────────────────
        // ── Step 5: take top 3 and build context string ───────────────────
// Budget: CTX_SIZE=2048, reserve 256 output + ~150 prompt overhead
// → ~1600 chars max for RAG context (safe at ~4 chars/token average)
        static constexpr int MAX_CONTEXT_CHARS = 1200;
        static constexpr int MAX_CHUNK_CHARS   = 450;

        int take = std::min((int)results.size(), 3);
        std::string context;

        for (int i = 0; i < take; ++i) {
            std::string chunk_text = results[i].text;

            // Truncate individual chunk if needed
            if ((int)chunk_text.size() > MAX_CHUNK_CHARS)
                chunk_text = chunk_text.substr(0, MAX_CHUNK_CHARS) + "...";

            std::string entry = "[" + std::to_string(i + 1) + "] "
                                + chunk_text + "\n\n";

            // Stop adding chunks if we'd exceed total budget
            if ((int)(context.size() + entry.size()) > MAX_CONTEXT_CHARS) {
                LOGI("RAG: context budget reached after %d chunks", i);
                break;
            }
            context += entry;
        }

        if (context.empty()) {
            g_bb.tool_result    = "No relevant content found in the document.";
            g_bb.tool_succeeded = false;
            logStatus("❌ [2/3] RAG: context empty after budget check");
            return NodeStatus::FAILURE;
        }

        LOGI("RAG: final context = %zu chars", context.size());
        g_bb.tool_result    = context;
        g_bb.tool_succeeded = true;
        logStatus("✅ [2/3] RAG: " + std::to_string(take) + " chunks retrieved ("
                  + std::to_string(context.size()) + " chars)");
        return NodeStatus::SUCCESS;
    }

    // ── Build index from scratch ──────────────────────────────────────────
    bool buildIndex() {
        logStatus("🟡 [2/3] RAG: chunking PDF...");
        auto chunks = load_and_chunk_pdf(g_pdf_path, 600, 100);

        if (chunks.empty()) {
            g_bb.tool_result    = "Could not extract text from the PDF. "
                                  "Make sure it is a text-based (not scanned) PDF.";
            g_bb.tool_succeeded = false;
            logStatus("❌ [2/3] RAG: no chunks extracted");
            return false;
        }

        logStatus("🟡 [2/3] RAG: embedding " +
                  std::to_string(chunks.size()) + " chunks...");

        std::vector<std::vector<float>> embeddings;
        embeddings.reserve(chunks.size());

        for (size_t i = 0; i < chunks.size(); ++i) {
            auto emb = embed_passage(chunks[i]);
            if (!emb.empty()) {
                embeddings.push_back(std::move(emb));
            } else {
                // Keep chunks and embeddings in sync: push a zero vector as placeholder
                // so chunk_index labels remain valid inside HNSW.
                LOGE("RAG: empty embedding for chunk %zu — using zero vector", i);
                embeddings.push_back(std::vector<float>(get_embedding_dim(), 0.0f));
            }

            if ((i + 1) % 5 == 0 || i + 1 == chunks.size()) {
                logStatus("🟡 [2/3] RAG: embedded " +
                          std::to_string(i + 1) + "/" +
                          std::to_string(chunks.size()));
            }
        }

        vector_store_build(embeddings, chunks);

        // Persist to disk so the next session skips this step
        if (!g_index_dir.empty()) {
            vector_store_save(g_index_dir);
            saveEmbeddingsToDisk(embeddings);
            logStatus("✅ [2/3] RAG: index saved to disk");
        }

        // Keep in memory for reranking during this session
        g_stored_embeddings = std::move(embeddings);

        logStatus("✅ [2/3] RAG: index built (" +
                  std::to_string(chunks.size()) + " chunks)");
        return true;
    }

    // ── Persist raw embeddings alongside the HNSW index ──────────────────
    // Simple binary format: [n:int32][dim:int32][n * dim * float32]
    void saveEmbeddingsToDisk(const std::vector<std::vector<float>>& embs) {
        if (g_index_dir.empty() || embs.empty()) return;
        std::string path = g_index_dir + "/embeddings.bin";
        std::ofstream f(path, std::ios::binary);
        if (!f) { LOGE("RAG: cannot write embeddings.bin"); return; }

        int n   = (int)embs.size();
        int dim = (int)embs[0].size();
        f.write(reinterpret_cast<const char*>(&n),   sizeof(int));
        f.write(reinterpret_cast<const char*>(&dim), sizeof(int));
        for (const auto& e : embs)
            f.write(reinterpret_cast<const char*>(e.data()), dim * sizeof(float));

        LOGI("RAG: saved %d embeddings (dim=%d) → %s", n, dim, path.c_str());
    }

    // ── Load raw embeddings from disk ─────────────────────────────────────
    void loadEmbeddingsFromDisk() {
        std::string path = g_index_dir + "/embeddings.bin";
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            LOGI("RAG: no embeddings.bin at %s — rerank will be skipped", path.c_str());
            return;
        }

        int n = 0, dim = 0;
        f.read(reinterpret_cast<char*>(&n),   sizeof(int));
        f.read(reinterpret_cast<char*>(&dim), sizeof(int));

        if (n <= 0 || dim <= 0) {
            LOGE("RAG: corrupt embeddings.bin (n=%d dim=%d)", n, dim);
            return;
        }

        g_stored_embeddings.assign(n, std::vector<float>(dim));
        for (auto& e : g_stored_embeddings)
            f.read(reinterpret_cast<char*>(e.data()), dim * sizeof(float));

        LOGI("RAG: loaded %d embeddings (dim=%d) from disk", n, dim);
    }
};

// ════════════════════════════════════════════════════════════════════════════
// NODE 3a – RefinedAnswerNode
// Tool path: passes query + tool_result to LLM and streams the reply.
// ════════════════════════════════════════════════════════════════════════════
class RefinedAnswerNode : public SyncActionNode {
public:
    RefinedAnswerNode(const std::string& name, const NodeConfig& cfg)
            : SyncActionNode(name, cfg) {}
    static BT::PortsList providedPorts() { return {}; }

    NodeStatus tick() override {
        // Guard: don't call LLM with empty or suspiciously short context
        if (g_bb.tool_result.empty()) {
            logStatus("❌ [3/3] Empty tool result — skipping LLM call");
            g_bb.final_answer = "No relevant content was found in the document.";
            return NodeStatus::FAILURE;
        }

        logStatus("🟡 [3/3] Generating refined answer...");
        std::string result = refined_answer(
                g_bb.user_query,
                g_bb.selected_tool,
                g_bb.tool_result,
                g_on_token
        );

        if (result.empty() || result[0] == '[') {
            logStatus("❌ [3/3] Generation failed — returning raw tool result");
            g_bb.final_answer = g_bb.tool_result;
            return NodeStatus::FAILURE;
        }
        g_bb.final_answer = result;
        logStatus("✅ [3/3] Done (" + std::to_string(result.size()) + " chars)");
        return NodeStatus::SUCCESS;
    }
};

// ════════════════════════════════════════════════════════════════════════════
// NODE 3b – GeneralAnswerNode
// No-tool path: LLM answers directly with streaming.
// ════════════════════════════════════════════════════════════════════════════
class GeneralAnswerNode : public SyncActionNode {
public:
    GeneralAnswerNode(const std::string& name, const NodeConfig& cfg)
            : SyncActionNode(name, cfg) {}
    static BT::PortsList providedPorts() { return {}; }

    NodeStatus tick() override {
        logStatus("🟡 [2/2] Generating general answer...");
        std::string result = general_answer(g_bb.user_query, g_on_token);

        if (result.empty() || result[0] == '[') {
            logStatus("❌ [2/2] Generation failed");
            g_bb.final_answer = "I couldn't generate a response. Please try again.";
            return NodeStatus::FAILURE;
        }
        g_bb.final_answer = result;
        logStatus("✅ [2/2] Done (" + std::to_string(result.size()) + " chars)");
        return NodeStatus::SUCCESS;
    }
};

// ════════════════════════════════════════════════════════════════════════════
// BEHAVIOR TREE XML
//
// RootFallback
// ├── ToolPath:   Intent → ShouldUseTool? → ToolExecutor → RefinedAnswer
// └── DirectPath: Intent → GeneralAnswer
//
// The BT Fallback tries ToolPath first. If ShouldUseTool returns FAILURE
// (no tool needed), the whole ToolPath sequence fails and DirectPath runs.
// ════════════════════════════════════════════════════════════════════════════
static const char* BT_XML = R"(
<root BTCPP_format="4">
  <BehaviorTree ID="ChatTree">
    <Sequence name="Root">
      <IntentNode/>
      <Fallback name="DispatchFallback">
        <Sequence name="ToolPath">
          <ShouldUseTool/>
          <ToolExecutor/>
          <RefinedAnswer/>
        </Sequence>
        <GeneralAnswer/>
      </Fallback>
    </Sequence>
  </BehaviorTree>
</root>
)";

// ════════════════════════════════════════════════════════════════════════════
// run_agent — public entry point called from JNI
// ════════════════════════════════════════════════════════════════════════════
void run_agent(
        const std::string& query,
        std::function<void(const std::string&)> on_token,
        std::function<void(const std::string&)> on_status)
{
    g_on_token  = on_token;
    g_on_status = on_status;

    g_bb.user_query = query;
    g_bb.reset();

    logStatus("🚀 Query: " + query);

    BehaviorTreeFactory factory;
    factory.registerNodeType<IntentNode>        ("IntentNode");
    factory.registerNodeType<ShouldUseToolNode> ("ShouldUseTool");
    factory.registerNodeType<ToolExecutorNode>  ("ToolExecutor");
    factory.registerNodeType<RefinedAnswerNode> ("RefinedAnswer");
    factory.registerNodeType<GeneralAnswerNode> ("GeneralAnswer");

    BT::Tree tree;
    try {
        tree = factory.createTreeFromText(BT_XML);
    } catch (const std::exception& ex) {
        LOGE("BT init failed: %s", ex.what());
        logStatus("❌ BT init failed: " + std::string(ex.what()));
        if (on_token) on_token("Sorry, internal error initialising the agent.");
        return;
    }

    tree.tickWhileRunning();
    logStatus("🏁 Done | " +
              (g_bb.final_answer.size() > 80
               ? g_bb.final_answer.substr(0, 80) + "..."
               : g_bb.final_answer));
}