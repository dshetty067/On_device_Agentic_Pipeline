#pragma once
#include "blackboard.h"
#include "tool_registry.h"
#include <string>

// ── Prompt Builder ─────────────────────────────────────────────────────────
// Centralised place for ALL prompt construction.
// Keeps node code clean and makes prompts easy to tune.
class PromptBuilder {
public:
    // ── Planner prompt ─────────────────────────────────────────────────────
    // Asks LLM: given query + available tools + context snippet, what to do?
    // Returns a prompt that expects one of:
    //   ACTION: use_tool
    //   ACTION: rag_answer
    //   ACTION: general_answer
    //   REASON: <one line>
    static std::string plannerPrompt(const AgentBlackboard& bb,
                                     const ToolRegistry& registry) {
        std::string tool_list;
        for (const auto& [name, def] : registry.allTools()) {
            tool_list += "  - " + name + ": " + def.description + "\n";
        }

        return chatmlStart("system",
                           "You are a task planner. Decide how to handle the user query.\n"
                           "Output EXACTLY two lines:\n"
                           "ACTION: <one of: use_tool | rag_answer | general_answer>\n"
                           "REASON: <one sentence explaining why>\n\n"
                           "Rules:\n"
                           "- use_tool: when the query needs real-time data, calculation, or a specific function\n"
                           "- rag_answer: when the query is about the uploaded PDF document\n"
                           "- general_answer: when the query is general knowledge not in the PDF\n\n"
                           "Available tools:\n" + tool_list
        ) + chatmlMsg("user",
                      "Query: " + bb.user_query + "\n"
                                                  "PDF loaded: " + (bb.pdf_loaded ? "yes" : "no") + "\n"
                                                                                                    "Context preview: " + bb.retrieved_context.substr(
                              0, std::min((size_t)200, bb.retrieved_context.size()))
        ) + "<|im_start|>assistant\n";
    }

    // ── Tool-use prompt ─────────────────────────────────────────────────────
    // Full message history + tool schema injected silently.
    // LLM must output {"tool_call": {"name": "...", "args": {...}}}
    static std::string toolUsePrompt(const AgentBlackboard& bb,
                                     const ToolRegistry& registry) {
        std::string sys =
                "You are a precise assistant with access to tools.\n" +
                registry.buildSystemPromptBlock() +
                "IMPORTANT: If you decide to call a tool, output ONLY the JSON. Nothing else.";

        std::string prompt = chatmlStart("system", sys);
        prompt += buildHistory(bb);
        prompt += chatmlMsg("user", bb.user_query);
        prompt += "<|im_start|>assistant\n";
        return prompt;
    }

    // ── RAG answer prompt ──────────────────────────────────────────────────
    // Strict: answer ONLY from context. No hallucination.
    static std::string ragAnswerPrompt(const AgentBlackboard& bb) {
        std::string context = truncate(bb.retrieved_context, 2000);

        std::string sys =
                "You are a strict document Q&A assistant.\n"
                "Answer ONLY using the CONTEXT below. Do NOT use outside knowledge.\n"
                "If the answer is not in the context, say exactly: "
                "\"This information is not in the document.\"\n"
                "Be concise. Quote the document when helpful.";

        std::string prompt = chatmlStart("system", sys);
        prompt += buildHistory(bb);
        prompt += chatmlMsg("user",
                            "CONTEXT FROM DOCUMENT:\n"
                            "─────────────────────\n" + context + "\n"
                                                                  "─────────────────────\n\n"
                                                                  "QUESTION: " + bb.user_query
        );
        prompt += "<|im_start|>assistant\n";
        return prompt;
    }

    // ── Tool result + follow-up prompt ─────────────────────────────────────
    // After a tool executes, feed result back and ask for final answer.
    static std::string toolResultPrompt(const AgentBlackboard& bb) {
        std::string sys =
                "You are a helpful assistant. A tool was called and returned a result.\n"
                "Use the tool result to answer the user's question directly and concisely.\n"
                "Do NOT call another tool. Do NOT output JSON. Just answer in plain text.";

        std::string prompt = chatmlStart("system", sys);
        prompt += buildHistory(bb); // includes prior tool call in history
        prompt += chatmlMsg("user",
                            "Tool result: " + bb.tool_result + "\n\n"
                                                               "Now answer the original question: " + bb.user_query
        );
        prompt += "<|im_start|>assistant\n";
        return prompt;
    }

    // ── General answer prompt ──────────────────────────────────────────────
    // Fallback: LLM uses its own knowledge, clearly labelled.
    static std::string generalAnswerPrompt(const AgentBlackboard& bb) {
        std::string sys =
                "You are a helpful assistant. Answer the user's question using your knowledge.\n"
                "Be honest about uncertainty. Be concise.";

        std::string prompt = chatmlStart("system", sys);
        prompt += buildHistory(bb);
        prompt += chatmlMsg("user", bb.user_query);
        prompt += "<|im_start|>assistant\n";
        return prompt;
    }

    // ── Validation prompt ──────────────────────────────────────────────────
    // Asks LLM to verify its own answer against the context.
    // Returns "VALID" or "INVALID: <reason>"
    static std::string validationPrompt(const std::string& question,
                                        const std::string& context,
                                        const std::string& answer) {
        return chatmlStart("system",
                           "You are a fact-checker. Given a question, its source context, and a proposed answer,\n"
                           "determine if the answer is supported by the context.\n"
                           "Respond with EXACTLY one of:\n"
                           "VALID\n"
                           "INVALID: <reason in one sentence>"
        ) + chatmlMsg("user",
                      "QUESTION: " + truncate(question, 200) + "\n\n"
                                                               "CONTEXT:\n" + truncate(context, 800) + "\n\n"
                                                                                                       "ANSWER: " + truncate(answer, 300)
        ) + "<|im_start|>assistant\n";
    }

private:
    static std::string chatmlStart(const std::string& role,
                                   const std::string& content) {
        return "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
    }

    static std::string chatmlMsg(const std::string& role,
                                 const std::string& content) {
        return "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
    }

    static std::string buildHistory(const AgentBlackboard& bb) {
        std::string out;
        for (const auto& msg : bb.message_history) {
            switch (msg.role) {
                case MsgRole::SYSTEM:
                    // Already in the system block; skip to avoid duplication
                    break;
                case MsgRole::USER:
                    out += chatmlMsg("user", msg.content);
                    break;
                case MsgRole::ASSISTANT:
                    out += chatmlMsg("assistant", msg.content);
                    break;
                case MsgRole::TOOL:
                    out += chatmlMsg("user",
                                     "[Tool: " + msg.tool_name + "]\nResult: " + msg.content);
                    break;
            }
        }
        return out;
    }

    static std::string truncate(const std::string& s, size_t max) {
        if (s.size() <= max) return s;
        std::string t = s.substr(0, max);
        // Cut at sentence boundary
        size_t last_dot = t.find_last_of(".!?");
        if (last_dot != std::string::npos && last_dot > max / 2)
            return t.substr(0, last_dot + 1);
        return t + "...";
    }
};