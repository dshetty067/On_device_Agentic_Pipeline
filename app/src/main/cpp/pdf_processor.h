#pragma once
#include <string>
#include <vector>


std::vector<std::string> load_and_chunk_pdf(
        const std::string& pdf_path,
        int chunk_size = 350,
        int overlap    = 100
);