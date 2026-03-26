#include <jni.h>
#include <string>
#include "llm_engine.h"
#include "embedding_engine.h"
#include "agent_bt.h"

// ── Load generation model ──────────────────────────────────────────────────
extern "C"
JNIEXPORT void JNICALL
Java_com_example_bt_1agent_MainActivity_loadModel(
        JNIEnv* env, jobject /*thiz*/, jstring path)
{
    const char* p = env->GetStringUTFChars(path, nullptr);
    load_model(p);
    env->ReleaseStringUTFChars(path, p);
}

// ── Load embedding model ───────────────────────────────────────────────────
extern "C"
JNIEXPORT void JNICALL
Java_com_example_bt_1agent_MainActivity_loadEmbeddingModel(
        JNIEnv* env, jobject /*thiz*/, jstring path)
{
    const char* p = env->GetStringUTFChars(path, nullptr);
    load_embedding_model(p);
    env->ReleaseStringUTFChars(path, p);
}

// ── Init tool registry ─────────────────────────────────────────────────────
// Must be called AFTER both models are loaded.
extern "C"
JNIEXPORT void JNICALL
Java_com_example_bt_1agent_MainActivity_initToolRegistry(
        JNIEnv* /*env*/, jobject /*thiz*/)
{
    init_tool_registry();
}

// ── Run RAG agent ──────────────────────────────────────────────────────────
extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_bt_1agent_MainActivity_generateText(
        JNIEnv* env, jobject thiz,
        jstring pdfPath, jstring storeDir, jstring prompt)
{
    const char* pdf   = env->GetStringUTFChars(pdfPath,  nullptr);
    const char* store = env->GetStringUTFChars(storeDir, nullptr);
    const char* text  = env->GetStringUTFChars(prompt,   nullptr);

    JavaVM* jvm;
    env->GetJavaVM(&jvm);
    jobject globalThiz = env->NewGlobalRef(thiz);

    jclass     cls          = env->GetObjectClass(thiz);
    jmethodID  streamMethod = env->GetMethodID(cls, "streamToken",  "(Ljava/lang/String;)V");
    jmethodID  statusMethod = env->GetMethodID(cls, "streamStatus", "(Ljava/lang/String;)V");

    std::string finalResult;

    auto onToken = [jvm, globalThiz, streamMethod, &finalResult](const std::string& piece) {
        JNIEnv* e = nullptr;
        jvm->AttachCurrentThread(&e, nullptr);
        jstring js = e->NewStringUTF(piece.c_str());
        e->CallVoidMethod(globalThiz, streamMethod, js);
        e->DeleteLocalRef(js);
        finalResult += piece;
    };

    auto onStatus = [jvm, globalThiz, statusMethod](const std::string& s) {
        JNIEnv* e = nullptr;
        jvm->AttachCurrentThread(&e, nullptr);
        jstring js = e->NewStringUTF(s.c_str());
        e->CallVoidMethod(globalThiz, statusMethod, js);
        e->DeleteLocalRef(js);
    };

    run_agent(pdf, store, text, onToken, onStatus);

    // Return the final validated answer from the blackboard
    const std::string& answer = get_blackboard().final_answer;
    finalResult = answer.empty() ? finalResult : answer;

    env->ReleaseStringUTFChars(pdfPath,  pdf);
    env->ReleaseStringUTFChars(storeDir, store);
    env->ReleaseStringUTFChars(prompt,   text);
    env->DeleteGlobalRef(globalThiz);

    return env->NewStringUTF(finalResult.c_str());
}