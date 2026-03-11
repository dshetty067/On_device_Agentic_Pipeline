#pragma once
#include <string>
#include <vector>

// Reads a PDF file using raw byte parsing (no Poppler needed)
// Returns list of text chunks (~500 chars each)
std::vector<std::string> load_and_chunk_pdf(
        const std::string& pdf_path,
        int chunk_size = 500,
        int overlap    = 50
);