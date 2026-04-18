#include <jni.h>
#include <string>
#include "llm_engine.h"
#include "embedding_engine.h"
#include "agent_bt.h"

// 🔥 declare setter
extern void agent_set_jni(JavaVM* jvm, jobject activity);

// ── Load generation model ────────────────────────────────
extern "C"
JNIEXPORT void JNICALL
Java_com_example_bt_1agent_MainActivity_loadModel(
        JNIEnv* env, jobject, jstring path)
{
    const char* p = env->GetStringUTFChars(path, nullptr);
    load_model(p);
    env->ReleaseStringUTFChars(path, p);
}

// ── Load embedding model ─────────────────────────────────
extern "C"
JNIEXPORT void JNICALL
Java_com_example_bt_1agent_MainActivity_loadEmbeddingModel(
        JNIEnv* env, jobject, jstring path)
{
    const char* p = env->GetStringUTFChars(path, nullptr);
    load_embedding_model(p);
    env->ReleaseStringUTFChars(path, p);
}

// ── Set PDF ──────────────────────────────────────────────
extern "C"
JNIEXPORT void JNICALL
Java_com_example_bt_1agent_MainActivity_setPdfPath(
        JNIEnv* env, jobject,
        jstring pdf_path, jstring index_dir)
{
    const char* pp = env->GetStringUTFChars(pdf_path,  nullptr);
    const char* id = env->GetStringUTFChars(index_dir, nullptr);

    agent_set_pdf_path(pp);
    agent_set_index_dir(id);

    env->ReleaseStringUTFChars(pdf_path,  pp);
    env->ReleaseStringUTFChars(index_dir, id);
}

// ── Run agent ────────────────────────────────────────────
extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_bt_1agent_MainActivity_generateText(
        JNIEnv* env, jobject thiz, jstring jquery)
{
    const char* q = env->GetStringUTFChars(jquery, nullptr);
    std::string query(q);
    env->ReleaseStringUTFChars(jquery, q);

    // 🔥 STORE JVM + ACTIVITY
    JavaVM* jvm;
    env->GetJavaVM(&jvm);
    jobject globalThiz = env->NewGlobalRef(thiz);

    agent_set_jni(jvm, globalThiz);

    jclass cls = env->GetObjectClass(thiz);
    jmethodID tok = env->GetMethodID(cls, "streamToken", "(Ljava/lang/String;)V");
    jmethodID sts = env->GetMethodID(cls, "streamStatus", "(Ljava/lang/String;)V");

    auto onToken = [jvm, globalThiz, tok](const std::string& piece) {
        JNIEnv* e = nullptr;
        jvm->AttachCurrentThread(&e, nullptr);

        jstring js = e->NewStringUTF(piece.c_str());
        e->CallVoidMethod(globalThiz, tok, js);
        e->DeleteLocalRef(js);
    };

    auto onStatus = [jvm, globalThiz, sts](const std::string& msg) {
        JNIEnv* e = nullptr;
        jvm->AttachCurrentThread(&e, nullptr);

        jstring js = e->NewStringUTF(msg.c_str());
        e->CallVoidMethod(globalThiz, sts, js);
        e->DeleteLocalRef(js);
    };

    run_agent(query, onToken, onStatus);

    return env->NewStringUTF("");
}