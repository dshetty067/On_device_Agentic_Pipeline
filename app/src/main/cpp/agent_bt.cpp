#include "agent_bt.h"
#include "blackboard.h"
#include "tool_registry.h"
#include "prompt_builder.h"
#include "llm_engine.h"
#include "pdf_processor.h"
#include "embedding_engine.h"
#include "vector_store.h"

#include <behaviortree_cpp/bt_factory.h>
#include <android/log.h>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <cmath>

#define LOG_TAG "BT_AGENT"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

using namespace BT;

// ── Shared state ───────────────────────────────────────────────────────────
static AgentBlackboard g_bb;

const AgentBlackboard& get_blackboard() { return g_bb; }

// ── Callback holders (set in run_agent, used by all nodes) ─────────────────
static std::function<void(const std::string&)> g_on_token;
static std::function<void(const std::string&)> g_on_status;

static void agentStatus(const std::string& msg) {
    LOGI("%s", msg.c_str());
    if (g_on_status) g_on_status(msg);
}

// ── String helpers ─────────────────────────────────────────────────────────
static std::string trimStr(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\n\r");
    size_t b = s.find_last_not_of(" \t\n\r");
    return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
}

static bool startsWith(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() &&
           s.substr(0, prefix.size()) == prefix;
}

static std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}

// ════════════════════════════════════════════════════════════════════════════
// NODE 1: PDF LOADER
// ════════════════════════════════════════════════════════════════════════════
class PDFLoaderNode : public SyncActionNode {
public:
    PDFLoaderNode(const std::string& name, const NodeConfig& cfg)
            : SyncActionNode(name, cfg) {}

    // FIX: BT.CPP v4 REQUIRES this static method on every node class.
    // Without it, registerNodeType<T>() cannot build the manifest and
    // VerifyXML will not find the node → throws → native crash.
    static BT::PortsList providedPorts() { return {}; }

    NodeStatus tick() override {
        if (g_bb.pdf_loaded && g_bb.embeddings_ready) {
            agentStatus("⚡ PDF already loaded — skipping");
            return NodeStatus::SUCCESS;
        }
        agentStatus("🟡 [1/8] Loading PDF: " + g_bb.pdf_path);
        auto chunks = load_and_chunk_pdf(g_bb.pdf_path);
        if (chunks.empty()) {
            agentStatus("❌ [1/8] PDF extraction failed");
            return NodeStatus::FAILURE;
        }
        g_bb.pdf_loaded = true;
        std::ostringstream oss;
        oss << "✅ [1/8] PDF loaded — " << chunks.size() << " chunks";
        agentStatus(oss.str());
        for (int i = 0; i < std::min((int)chunks.size(), 2); i++) {
            agentStatus("📄 Chunk " + std::to_string(i+1) + ": " +
                        chunks[i].substr(0, 80) + "...");
        }
        return NodeStatus::SUCCESS;
    }
};

// ════════════════════════════════════════════════════════════════════════════
// NODE 2: EMBED
// ════════════════════════════════════════════════════════════════════════════
class EmbedNode : public SyncActionNode {
public:
    EmbedNode(const std::string& name, const NodeConfig& cfg)
            : SyncActionNode(name, cfg) {}
    static BT::PortsList providedPorts() { return {}; }

    NodeStatus tick() override {
        if (g_bb.embeddings_ready) {
            agentStatus("⚡ [2/8] Embeddings already ready");
            return NodeStatus::SUCCESS;
        }
        agentStatus("🟡 [2/8] Checking vector store...");
        std::string bin = g_bb.store_dir + "/vector_store.bin";
        std::ifstream check(bin);
        if (check.good()) {
            check.close();
            agentStatus("⚡ [2/8] Loading saved vector store...");
            if (vector_store_load(g_bb.store_dir)) {
                g_bb.embeddings_ready = true;
                agentStatus("✅ [2/8] Vector store loaded from disk");
                return NodeStatus::SUCCESS;
            }
        }
        auto chunks = load_and_chunk_pdf(g_bb.pdf_path, 400, 80);
        if (chunks.empty()) {
            agentStatus("❌ [2/8] No chunks to embed");
            return NodeStatus::FAILURE;
        }
        agentStatus("🟡 [2/8] Embedding " + std::to_string(chunks.size()) + " chunks...");
        std::vector<std::vector<float>> embeddings;
        std::vector<std::string> embedded_chunks;
        embeddings.reserve(chunks.size());
        embedded_chunks.reserve(chunks.size());
        for (int i = 0; i < (int)chunks.size(); i++) {
            auto emb = embed_text("passage: " + chunks[i]);
            if (emb.empty()) continue;
            embeddings.push_back(emb);
            embedded_chunks.push_back(chunks[i]);
            if (i % 5 == 0 || i == (int)chunks.size()-1) {
                agentStatus("🔢 Embedded " + std::to_string(i+1) +
                            "/" + std::to_string(chunks.size()) + " chunks");
            }
        }
        vector_store_build(embeddings, embedded_chunks);
        vector_store_save(g_bb.store_dir);
        g_bb.embeddings_ready = true;
        agentStatus("✅ [2/8] Embeddings saved");
        return NodeStatus::SUCCESS;
    }
};

// ════════════════════════════════════════════════════════════════════════════
// NODE 3: RAG RETRIEVAL
// ════════════════════════════════════════════════════════════════════════════
class RetrieveDocumentsNode : public SyncActionNode {
public:
    RetrieveDocumentsNode(const std::string& name, const NodeConfig& cfg)
            : SyncActionNode(name, cfg) {}
    static BT::PortsList providedPorts() { return {}; }

    NodeStatus tick() override {
        agentStatus("🟡 [3/8] Retrieving relevant document sections...");
        LOGI("RetrieveDocumentsNode: query = %s", g_bb.user_query.c_str());
        auto query_emb = embed_text("query: " + g_bb.user_query);
        if (query_emb.empty()) {
            agentStatus("❌ [3/8] Query embedding failed");
            g_bb.retrieved_context = "";
            g_bb.best_relevance_score = 0.0f;
            return NodeStatus::FAILURE;
        }
        auto results = vector_store_search(query_emb, 5);
        if (results.empty()) {
            agentStatus("⚠️ [3/8] No results in vector store");
            g_bb.retrieved_context = "";
            g_bb.best_relevance_score = 0.0f;
            return NodeStatus::SUCCESS;
        }
        g_bb.best_relevance_score = results[0].score;
        if (g_bb.best_relevance_score >= 0.75f)
            g_bb.relevance_level = RelevanceLevel::HIGH;
        else if (g_bb.best_relevance_score >= 0.55f)
            g_bb.relevance_level = RelevanceLevel::MEDIUM;
        else if (g_bb.best_relevance_score >= 0.35f)
            g_bb.relevance_level = RelevanceLevel::LOW;
        else
            g_bb.relevance_level = RelevanceLevel::NONE;

        std::ostringstream ctx;
        for (int i = 0; i < (int)results.size(); i++) {
            std::string score_str = std::to_string(results[i].score).substr(0, 5);
            agentStatus("📌 [" + std::to_string(i+1) + "] score=" + score_str +
                        " → " + results[i].text.substr(0, 100) + "...");
            ctx << "[Context " << (i+1) << " | relevance=" << score_str << "]:\n"
                << results[i].text << "\n\n";
        }
        g_bb.retrieved_context = ctx.str();
        agentStatus("✅ [3/8] Retrieved " + std::to_string(results.size()) +
                    " chunks (best score=" +
                    std::to_string(g_bb.best_relevance_score).substr(0, 5) + ")");
        return NodeStatus::SUCCESS;
    }
};

// ════════════════════════════════════════════════════════════════════════════
// CONDITION: CheckRelevanceScore
//
// FIX: Must inherit from ConditionNode directly — NOT from a shared base
// that also inherits SyncActionNode. BT.CPP v4 registerNodeType<T>() reads
// the NodeType from the class hierarchy at compile time. A class that
// inherits ConditionNode correctly maps to NodeType::CONDITION in the
// manifest, which VerifyXML then accepts in Selector/Sequence positions
// where a Condition is expected.
// ════════════════════════════════════════════════════════════════════════════
class CheckRelevanceScoreNode : public ConditionNode {
public:
    CheckRelevanceScoreNode(const std::string& name, const NodeConfig& cfg)
            : ConditionNode(name, cfg) {}
    static BT::PortsList providedPorts() { return {}; }

    NodeStatus tick() override {
        bool relevant = (g_bb.relevance_level == RelevanceLevel::MEDIUM ||
                         g_bb.relevance_level == RelevanceLevel::HIGH);
        agentStatus(relevant
                    ? "✅ [4/8] Context relevance: sufficient (score=" +
                      std::to_string(g_bb.best_relevance_score).substr(0, 5) + ")"
                    : "⚠️ [4/8] Context relevance: low (score=" +
                      std::to_string(g_bb.best_relevance_score).substr(0, 5) +
                      ") — routing to fallback");
        return relevant ? NodeStatus::SUCCESS : NodeStatus::FAILURE;
    }
};

// ════════════════════════════════════════════════════════════════════════════
// NODE 4: PLANNER
// ════════════════════════════════════════════════════════════════════════════
class PlannerNode : public SyncActionNode {
public:
    PlannerNode(const std::string& name, const NodeConfig& cfg)
            : SyncActionNode(name, cfg) {}
    static BT::PortsList providedPorts() { return {}; }

    NodeStatus tick() override {
        agentStatus("🟡 [4/8] Planner: analysing query...");
        std::string q_lower = toLower(g_bb.user_query);
        static const std::vector<std::string> pdf_signals = {
                "document", "pdf", "file", "text", "says", "mention", "according",
                "paragraph", "section", "chapter", "page", "describe", "explain"
        };
        for (const auto& sig : pdf_signals) {
            if (q_lower.find(sig) != std::string::npos &&
                g_bb.relevance_level != RelevanceLevel::NONE) {
                g_bb.planned_action    = "rag_answer";
                g_bb.planner_reasoning = "Heuristic: query contains PDF signal word '" + sig + "'";
                agentStatus("⚡ [4/8] Planner (heuristic): rag_answer");
                return NodeStatus::SUCCESS;
            }
        }
        if (g_bb.relevance_level == RelevanceLevel::HIGH) {
            g_bb.planned_action    = "rag_answer";
            g_bb.planner_reasoning = "Heuristic: very high relevance score";
            agentStatus("⚡ [4/8] Planner (heuristic): rag_answer (high relevance)");
            return NodeStatus::SUCCESS;
        }
        std::string prompt = PromptBuilder::plannerPrompt(g_bb, getToolRegistry());
        std::string raw    = generate(prompt, nullptr);
        LOGI("PlannerNode raw output: %s", raw.c_str());

        std::istringstream ss(raw);
        std::string line;
        while (std::getline(ss, line)) {
            line = trimStr(line);
            if (startsWith(line, "ACTION:")) {
                std::string action = trimStr(line.substr(7));
                if (action == "use_tool" || action == "rag_answer" || action == "general_answer")
                    g_bb.planned_action = action;
                else
                    g_bb.planned_action = g_bb.pdf_loaded ? "rag_answer" : "general_answer";
            }
            if (startsWith(line, "REASON:"))
                g_bb.planner_reasoning = trimStr(line.substr(7));
        }
        if (g_bb.planned_action.empty()) {
            g_bb.planned_action    = g_bb.pdf_loaded ? "rag_answer" : "general_answer";
            g_bb.planner_reasoning = "Default routing (planner parse failed)";
        }
        agentStatus("✅ [4/8] Planner → " + g_bb.planned_action +
                    " | " + g_bb.planner_reasoning);
        return NodeStatus::SUCCESS;
    }
};

// ════════════════════════════════════════════════════════════════════════════
// CONDITION: ShouldUseTool
// FIX: ConditionNode — same reasoning as CheckRelevanceScoreNode above.
// ════════════════════════════════════════════════════════════════════════════
class ShouldUseToolNode : public ConditionNode {
public:
    ShouldUseToolNode(const std::string& name, const NodeConfig& cfg)
            : ConditionNode(name, cfg) {}
    static BT::PortsList providedPorts() { return {}; }

    NodeStatus tick() override {
        bool use_tool = (g_bb.planned_action == "use_tool");
        agentStatus(use_tool ? "🔧 Routing: tool execution path"
                             : "📚 Routing: answer path");
        return use_tool ? NodeStatus::SUCCESS : NodeStatus::FAILURE;
    }
};

// ════════════════════════════════════════════════════════════════════════════
// NODE 5a: LLM TOOL CALLER
// ════════════════════════════════════════════════════════════════════════════
class LLMToolCallerNode : public SyncActionNode {
public:
    LLMToolCallerNode(const std::string& name, const NodeConfig& cfg)
            : SyncActionNode(name, cfg) {}
    static BT::PortsList providedPorts() { return {}; }

    NodeStatus tick() override {
        if (g_bb.tool_call_count >= AgentBlackboard::MAX_TOOL_CALLS) {
            agentStatus("⚠️ Tool call limit reached — forcing direct answer");
            g_bb.planned_action = "rag_answer";
            return NodeStatus::FAILURE;
        }
        agentStatus("🟡 [5/8] LLM generating tool call...");
        std::string prompt = PromptBuilder::toolUsePrompt(g_bb, getToolRegistry());
        std::string raw    = generate(prompt, nullptr);
        LOGI("LLMToolCallerNode raw: %s", raw.c_str());
        g_bb.last_llm_output   = raw;
        g_bb.pending_tool_call = parseToolCall(raw);
        if (!g_bb.pending_tool_call.valid) {
            agentStatus("⚠️ [5/8] LLM did not produce a valid tool call — rerouting to answer");
            g_bb.planned_action = "rag_answer";
            return NodeStatus::FAILURE;
        }
        agentStatus("✅ [5/8] Tool call parsed: " + g_bb.pending_tool_call.tool_name);
        return NodeStatus::SUCCESS;
    }
};

// ════════════════════════════════════════════════════════════════════════════
// CONDITION: HasToolCall
// FIX: ConditionNode — same reasoning as CheckRelevanceScoreNode above.
// ════════════════════════════════════════════════════════════════════════════
class HasToolCallNode : public ConditionNode {
public:
    HasToolCallNode(const std::string& name, const NodeConfig& cfg)
            : ConditionNode(name, cfg) {}
    static BT::PortsList providedPorts() { return {}; }

    NodeStatus tick() override {
        return g_bb.pending_tool_call.valid
               ? NodeStatus::SUCCESS : NodeStatus::FAILURE;
    }
};

// ════════════════════════════════════════════════════════════════════════════
// NODE 5b: DISPATCH TOOL
// ════════════════════════════════════════════════════════════════════════════
class DispatchToolNode : public SyncActionNode {
public:
    DispatchToolNode(const std::string& name, const NodeConfig& cfg)
            : SyncActionNode(name, cfg) {}
    static BT::PortsList providedPorts() { return {}; }

    NodeStatus tick() override {
        const std::string& tool_name = g_bb.pending_tool_call.tool_name;
        agentStatus("🔧 [6/8] Executing tool: " + tool_name);
        const ToolDefinition* def = getToolRegistry().getTool(tool_name);
        if (!def) {
            agentStatus("❌ [6/8] Unknown tool: " + tool_name);
            g_bb.tool_result    = "Error: tool '" + tool_name + "' not found";
            g_bb.tool_succeeded = false;
            return NodeStatus::FAILURE;
        }
        g_bb.tool_call_count++;
        try {
            g_bb.tool_result    = def->executor(g_bb.pending_tool_call.args);
            g_bb.tool_succeeded = true;
            agentStatus("✅ [6/8] Tool result: " +
                        g_bb.tool_result.substr(0, std::min((size_t)100, g_bb.tool_result.size())));
            g_bb.appendHistory(MsgRole::TOOL, g_bb.tool_result, tool_name);
            return NodeStatus::SUCCESS;
        } catch (const std::exception& ex) {
            g_bb.tool_result    = "Tool error: " + std::string(ex.what());
            g_bb.tool_succeeded = false;
            agentStatus("❌ [6/8] Tool threw: " + std::string(ex.what()));
            return NodeStatus::FAILURE;
        }
    }
};

// ════════════════════════════════════════════════════════════════════════════
// NODE 6: RAG ANSWER
// ════════════════════════════════════════════════════════════════════════════
class RAGAnswerNode : public SyncActionNode {
public:
    RAGAnswerNode(const std::string& name, const NodeConfig& cfg)
            : SyncActionNode(name, cfg) {}
    static BT::PortsList providedPorts() { return {}; }

    NodeStatus tick() override {
        agentStatus("🟡 [6/8] Generating answer from document context...");
        std::string prompt;
        if (g_bb.tool_succeeded && !g_bb.tool_result.empty())
            prompt = PromptBuilder::toolResultPrompt(g_bb);
        else
            prompt = PromptBuilder::ragAnswerPrompt(g_bb);

        std::string result = generate(prompt, g_on_token);
        if (result.empty() || result[0] == '[') {
            agentStatus("❌ [6/8] LLM generation failed");
            return NodeStatus::FAILURE;
        }
        auto pos = result.find("<|im_end|>");
        if (pos != std::string::npos) result.erase(pos);
        g_bb.last_llm_output = result;
        g_bb.final_answer    = trimStr(result);
        agentStatus("✅ [6/8] RAG answer generated (" +
                    std::to_string(g_bb.final_answer.size()) + " chars)");
        return NodeStatus::SUCCESS;
    }
};

// ════════════════════════════════════════════════════════════════════════════
// NODE 7: VALIDATE RESULT
// ════════════════════════════════════════════════════════════════════════════
class ValidateResultNode : public SyncActionNode {
public:
    ValidateResultNode(const std::string& name, const NodeConfig& cfg)
            : SyncActionNode(name, cfg) {}
    static BT::PortsList providedPorts() { return {}; }

    NodeStatus tick() override {
        if (g_bb.retrieved_context.empty() || g_bb.final_answer.empty()) {
            g_bb.answer_validated  = true;
            g_bb.validation_reason = "No context to validate against";
            agentStatus("⚡ [7/8] Validation skipped (no context)");
            return NodeStatus::SUCCESS;
        }
        agentStatus("🟡 [7/8] Validating answer against context...");
        std::string prompt = PromptBuilder::validationPrompt(
                g_bb.user_query, g_bb.retrieved_context, g_bb.final_answer);
        std::string verdict = trimStr(generate(prompt, nullptr));
        LOGI("ValidateResultNode verdict: %s", verdict.c_str());

        if (startsWith(verdict, "VALID")) {
            g_bb.answer_validated  = true;
            g_bb.validation_reason = "Self-verified against context";
            agentStatus("✅ [7/8] Answer validated");
        } else {
            g_bb.answer_validated  = false;
            g_bb.validation_reason = verdict;
            std::string flagged = g_bb.final_answer +
                                  "\n\n⚠️ [Note: Confidence check flagged: " +
                                  verdict.substr(std::min((size_t)9, verdict.size())) + "]";
            g_bb.final_answer = flagged;
            agentStatus("⚠️ [7/8] Validation: " + verdict);
        }
        return NodeStatus::SUCCESS;
    }
};

// ════════════════════════════════════════════════════════════════════════════
// NODE 8: GENERAL ANSWER
// ════════════════════════════════════════════════════════════════════════════
class GeneralAnswerNode : public SyncActionNode {
public:
    GeneralAnswerNode(const std::string& name, const NodeConfig& cfg)
            : SyncActionNode(name, cfg) {}
    static BT::PortsList providedPorts() { return {}; }

    NodeStatus tick() override {
        agentStatus("🟡 [8/8] General knowledge answer (no document match)...");
        std::string prompt = PromptBuilder::generalAnswerPrompt(g_bb);
        std::string result = generate(prompt, g_on_token);
        if (result.empty() || result[0] == '[') {
            agentStatus("❌ [8/8] LLM generation failed");
            g_bb.final_answer = "I was unable to generate an answer. Please try rephrasing.";
            return NodeStatus::FAILURE;
        }
        auto pos = result.find("<|im_end|>");
        if (pos != std::string::npos) result.erase(pos);
        g_bb.last_llm_output  = result;
        g_bb.final_answer     = trimStr(result);
        g_bb.answer_validated  = true;
        g_bb.validation_reason = "General knowledge — not document-verified";
        agentStatus("✅ [8/8] General answer generated");
        return NodeStatus::SUCCESS;
    }
};

// ════════════════════════════════════════════════════════════════════════════
// NODE: HISTORY MANAGER
// ════════════════════════════════════════════════════════════════════════════
class HistoryManagerNode : public SyncActionNode {
public:
    HistoryManagerNode(const std::string& name, const NodeConfig& cfg)
            : SyncActionNode(name, cfg) {}
    static BT::PortsList providedPorts() { return {}; }

    NodeStatus tick() override {
        g_bb.appendHistory(MsgRole::USER,      g_bb.user_query);
        g_bb.appendHistory(MsgRole::ASSISTANT, g_bb.final_answer);
        LOGI("HistoryManager: history size = %zu", g_bb.message_history.size());
        return NodeStatus::SUCCESS;
    }
};

// ════════════════════════════════════════════════════════════════════════════
// TOOL REGISTRY — unchanged from original
// ════════════════════════════════════════════════════════════════════════════
void init_tool_registry() {
    auto& reg = getToolRegistry();

    reg.registerTool({
                             "search_pdf",
                             "Search the uploaded PDF document for relevant passages on a specific topic",
                             {{"query", "string", "The search query to look up in the document", true}},
                             [](const std::map<std::string, std::string>& args) -> std::string {
                                 auto it = args.find("query");
                                 if (it == args.end()) return "Error: missing query parameter";
                                 auto emb = embed_text("query: " + it->second);
                                 if (emb.empty()) return "Error: embedding failed";
                                 auto results = vector_store_search(emb, 3);
                                 if (results.empty()) return "No relevant passages found for: " + it->second;
                                 std::ostringstream oss;
                                 for (int i = 0; i < (int)results.size(); i++) {
                                     oss << "[Result " << (i+1) << " | score="
                                         << std::to_string(results[i].score).substr(0, 5) << "]:\n"
                                         << results[i].text << "\n\n";
                                 }
                                 g_bb.retrieved_context = oss.str();
                                 return oss.str();
                             }
                     });

    reg.registerTool({
                             "calculate",
                             "Evaluate a simple arithmetic expression (e.g. '2 + 3 * 4', '100 / 5')",
                             {{"expression", "string", "The arithmetic expression to evaluate", true}},
                             [](const std::map<std::string, std::string>& args) -> std::string {
                                 auto it = args.find("expression");
                                 if (it == args.end()) return "Error: missing expression";
                                 const std::string& expr = it->second;
                                 for (char c : expr) {
                                     if (!std::isdigit(c) && c != '+' && c != '-' && c != '*' &&
                                         c != '/' && c != '.' && c != ' ' && c != '(' && c != ')') {
                                         return "Error: unsupported character '" + std::string(1, c) +
                                                "' — only basic arithmetic supported";
                                     }
                                 }
                                 double a = 0, b = 0; char op = 0;
                                 if (std::sscanf(expr.c_str(), "%lf %c %lf", &a, &op, &b) == 3) {
                                     double result = 0;
                                     switch (op) {
                                         case '+': result = a + b; break;
                                         case '-': result = a - b; break;
                                         case '*': result = a * b; break;
                                         case '/':
                                             if (b == 0) return "Error: division by zero";
                                             result = a / b; break;
                                         default: return "Error: unsupported operator '" + std::string(1, op) + "'";
                                     }
                                     std::ostringstream oss;
                                     oss << expr << " = " << result;
                                     return oss.str();
                                 }
                                 return "Error: could not parse expression '" + expr + "'";
                             }
                     });

    reg.registerTool({
                             "summarise_document",
                             "Produce a high-level summary of the entire uploaded PDF document",
                             {},
                             [](const std::map<std::string, std::string>&) -> std::string {
                                 auto emb = embed_text("query: overview summary introduction conclusion");
                                 if (emb.empty()) return "Error: could not embed summary query";
                                 auto results = vector_store_search(emb, 5);
                                 if (results.empty()) return "No document content available to summarise";
                                 std::ostringstream oss;
                                 oss << "Document excerpt (for summarisation):\n";
                                 for (int i = 0; i < (int)results.size(); i++)
                                     oss << results[i].text << "\n\n";
                                 g_bb.retrieved_context = oss.str();
                                 return oss.str();
                             }
                     });

    LOGI("ToolRegistry: %zu tools registered", getToolRegistry().allTools().size());
}

// ════════════════════════════════════════════════════════════════════════════
// BT XML
// ════════════════════════════════════════════════════════════════════════════
static const char* BT_XML = R"(
<root BTCPP_format="4">
  <BehaviorTree ID="AgentTree">
    <Fallback name="RootSelector">

      <!-- Tool path -->
      <Sequence name="ToolPath">
        <PDFLoader/>
        <Embed/>
        <RetrieveDocuments/>
        <Planner/>
        <ShouldUseTool/>
        <LLMToolCaller/>
        <HasToolCall/>
        <DispatchTool/>
        <RAGAnswer/>
        <ValidateResult/>
        <HistoryManager/>
      </Sequence>

      <!-- RAG path -->
      <Sequence name="RAGPath">
        <PDFLoader/>
        <Embed/>
        <RetrieveDocuments/>
        <Fallback name="AnswerStrategy">

          <Sequence name="RAGSequence">
            <CheckRelevanceScore/>
            <RAGAnswer/>
            <ValidateResult/>
          </Sequence>

          <GeneralAnswer/>

        </Fallback>
        <HistoryManager/>
      </Sequence>

    </Fallback>
  </BehaviorTree>
</root>
)";

// ════════════════════════════════════════════════════════════════════════════
// run_agent — entry point called from JNI
//
// FIX: Use registerNodeType<T>() instead of the raw registerBuilder<> lambda.
//
// registerNodeType<T>() is the correct BT.CPP v4 API. It:
//   1. Calls T::providedPorts() to build the manifest
//   2. Reads the NodeType from the class hierarchy (Action vs Condition)
//   3. Registers both in the factory's manifest map
//
// VerifyXML checks the manifest map — if a node appears in the XML but
// has no manifest entry, or the NodeType in the manifest doesn't match
// how the XML uses it (e.g. as a leaf vs a control node), it throws
// RuntimeError → __cxa_throw → unhandled → native crash (SIGABRT).
//
// The old raw registerBuilder<> lambda bypassed manifest registration,
// so VerifyXML could not find the node entries → crash.
// ════════════════════════════════════════════════════════════════════════════
void run_agent(
        const std::string& pdf_path,
        const std::string& store_dir,
        const std::string& query,
        std::function<void(const std::string&)> on_token,
        std::function<void(const std::string&)> on_status)
{
    g_on_token  = on_token;
    g_on_status = on_status;

    g_bb.pdf_path   = pdf_path;
    g_bb.store_dir  = store_dir;
    g_bb.user_query = query;
    g_bb.reset_per_turn();

    agentStatus("🚀 Agent start | query: " + query);

    BehaviorTreeFactory factory;

    // ── Action nodes ───────────────────────────────────────────────────────
    // registerNodeType<T>() is the v4-correct API. It registers the node's
    // manifest (ports + NodeType) AND its builder in one call.
    factory.registerNodeType<PDFLoaderNode>        ("PDFLoader");
    factory.registerNodeType<EmbedNode>            ("Embed");
    factory.registerNodeType<RetrieveDocumentsNode>("RetrieveDocuments");
    factory.registerNodeType<PlannerNode>          ("Planner");
    factory.registerNodeType<LLMToolCallerNode>    ("LLMToolCaller");
    factory.registerNodeType<DispatchToolNode>      ("DispatchTool");
    factory.registerNodeType<RAGAnswerNode>        ("RAGAnswer");
    factory.registerNodeType<ValidateResultNode>   ("ValidateResult");
    factory.registerNodeType<GeneralAnswerNode>    ("GeneralAnswer");
    factory.registerNodeType<HistoryManagerNode>   ("HistoryManager");

    // ── Condition nodes ────────────────────────────────────────────────────
    // Same API — registerNodeType<T>() detects ConditionNode in the hierarchy
    // and sets NodeType::CONDITION in the manifest automatically.
    factory.registerNodeType<ShouldUseToolNode>       ("ShouldUseTool");
    factory.registerNodeType<HasToolCallNode>          ("HasToolCall");
    factory.registerNodeType<CheckRelevanceScoreNode>  ("CheckRelevanceScore");

    // ── Build tree with error surfacing ────────────────────────────────────
    BT::Tree tree;
    try {
        tree = factory.createTreeFromText(BT_XML);
    } catch (const std::exception& ex) {
        // Surface the exact VerifyXML error message so it appears in logcat
        // instead of becoming an opaque native crash.
        LOGE("BT createTreeFromText FAILED: %s", ex.what());
        agentStatus("❌ BT init failed: " + std::string(ex.what()));
        return;
    }

    tree.tickWhileRunning();

    agentStatus("🏁 Agent done | answer: " +
                g_bb.final_answer.substr(0, std::min((size_t)80, g_bb.final_answer.size())));
}