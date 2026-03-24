#include "pdf_processor.h"
#include <fpdfview.h>
#include <fpdf_text.h>
#include <android/log.h>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>

#define LOG_TAG "PDFProcessor"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// ── PDFium init guard ────────────────────────────────────────────────────────
static void ensure_pdfium_init() {
    static bool initialised = false;
    if (!initialised) {
        FPDF_InitLibrary();
        initialised = true;
        LOGI("PDFium initialised");
    }
}

// ── UTF-16LE → UTF-8 helper ──────────────────────────────────────────────────
static std::string utf16le_to_utf8(const unsigned short* buf, int len) {
    std::string out;
    out.reserve(len);
    for (int i = 0; i < len; ++i) {
        unsigned int cp = buf[i];
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }
    return out;
}

// ── Core extractor ───────────────────────────────────────────────────────────
static std::string extract_text_from_pdf(const std::string& path) {
    ensure_pdfium_init();

    FPDF_DOCUMENT doc = FPDF_LoadDocument(path.c_str(), nullptr);
    if (!doc) {
        unsigned long err = FPDF_GetLastError();
        LOGE("PDFium failed to open '%s', error=%lu", path.c_str(), err);
        return "";
    }

    int page_count = FPDF_GetPageCount(doc);
    LOGI("PDF '%s': %d pages", path.c_str(), page_count);

    std::string full_text;

    for (int pi = 0; pi < page_count; ++pi) {
        FPDF_PAGE page = FPDF_LoadPage(doc, pi);
        if (!page) {
            LOGE("Cannot load page %d", pi);
            continue;
        }

        FPDF_TEXTPAGE tp = FPDFText_LoadPage(page);
        if (!tp) {
            LOGE("Cannot load text page %d", pi);
            FPDF_ClosePage(page);
            continue;
        }

        int char_count = FPDFText_CountChars(tp);
        LOGI("Page %d: %d chars", pi, char_count);

        if (char_count > 0) {
            // GetText wants UTF-16LE; buffer needs char_count+1 code units
            std::vector<unsigned short> buf(char_count + 1, 0);
            FPDFText_GetText(tp, 0, char_count, buf.data());
            std::string page_text = utf16le_to_utf8(buf.data(), char_count);

            // Replace form-feeds / PDF paragraph separators with spaces
            for (char& c : page_text) {
                if (c == '\r' || c == '\f') c = '\n';
            }

            full_text += page_text;
            full_text += '\n'; // page separator
        }

        FPDFText_ClosePage(tp);
        FPDF_ClosePage(page);
    }

    FPDF_CloseDocument(doc);
    LOGI("Total extracted: %zu chars across %d pages",
         full_text.size(), page_count);
    return full_text;
}

// ── Public API ───────────────────────────────────────────────────────────────
std::vector<std::string> load_and_chunk_pdf(
        const std::string& pdf_path,
        int chunk_size,
        int overlap)
{
    LOGI("load_and_chunk_pdf: '%s'  chunk=%d overlap=%d",
         pdf_path.c_str(), chunk_size, overlap);

    std::string text = extract_text_from_pdf(pdf_path);
    if (text.empty()) {
        LOGE("No text extracted from PDF");
        return {};
    }

    // ── Normalise whitespace ─────────────────────────────────────────────────
    std::string cleaned;
    cleaned.reserve(text.size());
    bool last_space = false;
    for (unsigned char c : text) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            if (!last_space) { cleaned += ' '; last_space = true; }
        } else {
            cleaned += static_cast<char>(c);
            last_space = false;
        }
    }

    // ── Chunk with overlap ───────────────────────────────────────────────────
    std::vector<std::string> chunks;
    int len   = static_cast<int>(cleaned.size());
    int step  = chunk_size - overlap;
    if (step <= 0) {
        LOGE("Invalid chunk_size/overlap: %d/%d", chunk_size, overlap);
        return {};
    }

    for (int start = 0; start < len; start += step) {
        int end = std::min(start + chunk_size, len);
        chunks.emplace_back(cleaned.substr(start, end - start));
        LOGI("Chunk %zu: [%d, %d) = %d chars",
             chunks.size(), start, end, end - start);
    }

    LOGI("Total chunks: %zu", chunks.size());
    return chunks;
}