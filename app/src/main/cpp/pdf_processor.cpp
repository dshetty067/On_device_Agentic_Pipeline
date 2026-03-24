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

    // ── Assemble sentence-aware chunks ──────────────────────────────────────
    std::vector<std::string> chunks;
    std::string cur_chunk;
    cur_chunk.reserve(chunk_size + 200);

    // overlap_sentences: keep track of the last few sentences for the next chunk
    std::vector<std::string> overlap_buf;

    for (size_t si = 0; si < sentences.size(); ++si) {
        const std::string& sent = sentences[si];

        if (cur_chunk.empty()) {
            cur_chunk = sent;
        } else {
            cur_chunk += ' ';
            cur_chunk += sent;
        }

        // When chunk is big enough, flush it
        if ((int)cur_chunk.size() >= chunk_size) {
            chunks.push_back(cur_chunk);
            LOGI("Chunk %zu: %zu chars", chunks.size(), cur_chunk.size());

            // Build overlap: collect sentences from the end of cur_chunk
            // until we have ~overlap chars worth
            overlap_buf.clear();
            int acc = 0;
            // Walk backwards through sentences that contributed to this chunk
            // Simple approach: rewind si by enough sentences to cover overlap chars
            int rewind = 0;
            for (int back = (int)si; back >= 0 && acc < overlap; --back) {
                acc += (int)sentences[back].size() + 1;
                ++rewind;
            }
            // Start next chunk from (si - rewind + 1)
            size_t restart = (si >= (size_t)rewind) ? si - rewind + 1 : 0;
            cur_chunk.clear();
            for (size_t r = restart; r <= si; ++r) {
                if (!cur_chunk.empty()) cur_chunk += ' ';
                cur_chunk += sentences[r];
            }
        }
    }

    // flush the last partial chunk
    if (!cur_chunk.empty()) {
        chunks.push_back(cur_chunk);
        LOGI("Chunk %zu (final): %zu chars", chunks.size(), cur_chunk.size());
    }

    LOGI("Total chunks: %zu", chunks.size());
    return chunks;
}