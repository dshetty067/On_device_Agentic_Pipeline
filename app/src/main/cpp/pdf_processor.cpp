#include "pdf_processor.h"
#include <fstream>
#include <sstream>
#include <regex>
#include <android/log.h>

#define LOG_TAG "PDFProcessor"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Minimal PDF text extractor — reads raw bytes and pulls out text
// from BT (text block) stream operators. Works for most simple PDFs.
static std::string extract_text_from_pdf(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        LOGE("Cannot open PDF: %s", path.c_str());
        return "";
    }

    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    file.close();

    LOGI("PDF raw size: %zu bytes", content.size());

    // Extract text between BT...ET blocks (PDF text objects)
    std::string text;
    size_t pos = 0;

    while (pos < content.size()) {
        // Find BT (Begin Text)
        size_t bt = content.find("BT", pos);
        if (bt == std::string::npos) break;

        // Find matching ET (End Text)
        size_t et = content.find("ET", bt);
        if (et == std::string::npos) break;

        std::string block = content.substr(bt, et - bt);

        // Find all (text) Tj and [(text)] TJ operators
        size_t p = 0;
        while (p < block.size()) {
            if (block[p] == '(') {
                size_t end = p + 1;
                while (end < block.size() && block[end] != ')') {
                    if (block[end] == '\\') end++; // skip escaped
                    end++;
                }
                if (end < block.size()) {
                    std::string word = block.substr(p + 1, end - p - 1);
                    // Filter non-printable chars
                    std::string clean;
                    for (char c : word) {
                        if (c >= 32 && c < 127) clean += c;
                        else if (c == '\n' || c == '\r') clean += ' ';
                    }
                    if (!clean.empty()) {
                        text += clean + " ";
                    }
                }
                p = end + 1;
            } else {
                p++;
            }
        }

        pos = et + 2;
    }

    LOGI("Extracted text length: %zu chars", text.size());
    return text;
}

std::vector<std::string> load_and_chunk_pdf(
        const std::string& pdf_path,
        int chunk_size,
        int overlap)
{
    LOGI("load_and_chunk_pdf: %s", pdf_path.c_str());

    std::string text = extract_text_from_pdf(pdf_path);

    if (text.empty()) {
        LOGE("No text extracted from PDF");
        return {};
    }

    // Clean up whitespace
    std::string cleaned;
    bool last_space = false;
    for (char c : text) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            if (!last_space) cleaned += ' ';
            last_space = true;
        } else {
            cleaned += c;
            last_space = false;
        }
    }

    // Chunk with overlap
    std::vector<std::string> chunks;
    int len = (int)cleaned.size();
    int start = 0;

    while (start < len) {
        int end = std::min(start + chunk_size, len);
        chunks.push_back(cleaned.substr(start, end - start));
        LOGI("Chunk %zu: %d chars", chunks.size(), end - start);
        start += (chunk_size - overlap);
    }

    LOGI("Total chunks: %zu", chunks.size());
    return chunks;
}