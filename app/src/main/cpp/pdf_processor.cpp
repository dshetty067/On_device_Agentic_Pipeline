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
            std::vector<unsigned short> buf(char_count + 1, 0);
            FPDFText_GetText(tp, 0, char_count, buf.data());
            std::string page_text = utf16le_to_utf8(buf.data(), char_count);

            for (char& c : page_text) {
                if (c == '\r' || c == '\f') c = '\n';
            }

            full_text += page_text;
            full_text += '\n';
        }

        FPDFText_ClosePage(tp);
        FPDF_ClosePage(page);
    }

    FPDF_CloseDocument(doc);
    LOGI("Total extracted: %zu chars across %d pages", full_text.size(), page_count);
    return full_text;
}

// ── Sentence splitter ────────────────────────────────────────────────────────
// Splits text into sentences by . ! ? followed by space/newline or end.
// Keeps the delimiter attached to the sentence.
static std::vector<std::string> split_sentences(const std::string& text) {
    std::vector<std::string> sentences;
    std::string cur;
    cur.reserve(256);

    for (size_t i = 0; i < text.size(); ++i) {
        char c = text[i];
        cur += c;

        bool is_terminal = (c == '.' || c == '!' || c == '?');
        if (is_terminal) {
            // peek ahead — must be followed by space/newline or end of string
            size_t j = i + 1;
            while (j < text.size() && text[j] == ' ') ++j;
            bool followed_by_space_or_end =
                    (j >= text.size()) ||
                    (text[j] == '\n')  ||
                    (j > i + 1);       // there was at least one space after the dot

            if (followed_by_space_or_end && cur.size() >= 20) {
                // trim trailing whitespace
                size_t end = cur.find_last_not_of(" \t\n\r");
                if (end != std::string::npos)
                    sentences.push_back(cur.substr(0, end + 1));
                cur.clear();
                // skip the spaces we peeked past
                i = j - 1;
            }
        } else if (c == '\n') {
            // double newline = paragraph break → force a split
            if (i + 1 < text.size() && text[i + 1] == '\n') {
                size_t end = cur.find_last_not_of(" \t\n\r");
                if (end != std::string::npos && cur.size() >= 20)
                    sentences.push_back(cur.substr(0, end + 1));
                cur.clear();
            }
        }
    }

    // flush remainder
    size_t end = cur.find_last_not_of(" \t\n\r");
    if (end != std::string::npos && cur.size() >= 10)
        sentences.push_back(cur.substr(0, end + 1));

    return sentences;
}

// ── Public API ───────────────────────────────────────────────────────────────
// chunk_size  = target chars per chunk  (default 600)
// overlap     = how many chars of the previous chunk to repeat (default 100)
//
// Strategy: accumulate whole sentences until the chunk reaches chunk_size,
// then start a new chunk that begins overlap chars back (re-using tail of
// previous chunk). This avoids splitting mid-sentence and gives the embedding
// model complete, coherent text.
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

    // ── Normalise whitespace (collapse runs, keep single newlines) ───────────
    std::string cleaned;
    cleaned.reserve(text.size());
    bool last_ws = false;
    for (unsigned char c : text) {
        if (c == '\t' || c == '\r' || c == '\f') c = ' ';
        if (c == ' ') {
            if (!last_ws) { cleaned += ' '; last_ws = true; }
        } else {
            cleaned += static_cast<char>(c);
            last_ws = false;
        }
    }

    // ── Split into sentences ─────────────────────────────────────────────────
    std::vector<std::string> sentences = split_sentences(cleaned);
    LOGI("Sentences extracted: %zu", sentences.size());

    if (sentences.empty()) {
        // Fallback: raw character chunking if no sentences detected
        LOGI("No sentences found, falling back to raw chunking");
        std::vector<std::string> chunks;
        int len  = (int)cleaned.size();
        int step = chunk_size - overlap;
        if (step <= 0) step = chunk_size;
        for (int start = 0; start < len; start += step) {
            int end = std::min(start + chunk_size, len);
            chunks.emplace_back(cleaned.substr(start, end - start));
        }
        return chunks;
    }

    // ── Assemble sentence-aware chunks (FIXED VERSION) ──────────────────────
    std::vector<std::string> chunks;
    std::vector<std::string> window;

    int current_len = 0;

    for (const auto& sent : sentences) {
        window.push_back(sent);
        current_len += sent.size();

        if (current_len >= chunk_size) {
            // Build chunk
            std::string chunk;
            for (auto& s : window) {
                if (!chunk.empty()) chunk += " ";
                chunk += s;
            }
            chunks.push_back(chunk);

            LOGI("Chunk %zu: %zu chars", chunks.size(), chunk.size());

            // Build overlap (CORRECT way)
            std::vector<std::string> new_window;
            int acc = 0;

            for (int i = (int)window.size() - 1; i >= 0; --i) {
                acc += window[i].size();
                new_window.insert(new_window.begin(), window[i]);
                if (acc >= overlap) break;
            }

            window = new_window;

            // recalc length
            current_len = 0;
            for (auto& s : window) current_len += s.size();
        }
    }

// last chunk
    if (!window.empty()) {
        std::string chunk;
        for (auto& s : window) {
            if (!chunk.empty()) chunk += " ";
            chunk += s;
        }
        chunks.push_back(chunk);

        LOGI("Chunk %zu (final): %zu chars", chunks.size(), chunk.size());
    }

    LOGI("Total chunks: %zu", chunks.size());
    return chunks;
}