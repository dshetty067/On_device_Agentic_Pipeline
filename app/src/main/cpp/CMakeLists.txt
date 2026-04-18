cmake_minimum_required(VERSION 3.22.1)
project(bt_agent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_ANDROID_STL_TYPE c++_shared)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")


if(ANDROID_ABI STREQUAL "arm64-v8a")
    set(ARCH_FLAGS "-march=armv8-a -mtune=cortex-a73")
elseif(ANDROID_ABI STREQUAL "x86_64")
    set(ARCH_FLAGS "-mavx2 -mfma -mf16c")
else()
    set(ARCH_FLAGS "")
endif()


set(SAFE_MATH_FLAGS "-fno-math-errno -fno-trapping-math -fassociative-math -freciprocal-math -fno-signed-zeros -fno-finite-math-only")

set(CMAKE_C_FLAGS_RELEASE   "-O3 -DNDEBUG ${ARCH_FLAGS} ${SAFE_MATH_FLAGS} -funroll-loops")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG ${ARCH_FLAGS} ${SAFE_MATH_FLAGS} -funroll-loops")

set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -O3 ${ARCH_FLAGS} ${SAFE_MATH_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 ${ARCH_FLAGS} ${SAFE_MATH_FLAGS} -stdlib=libc++")

# ══════════════════════════════════════════════════════════════════════════════
# GGML / LLAMA.CPP BUILD FLAGS
# ══════════════════════════════════════════════════════════════════════════════

set(LLAMA_STATIC        ON  CACHE BOOL "" FORCE)

set(GGML_CUDA           OFF CACHE BOOL "" FORCE)
set(GGML_METAL          OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP         OFF CACHE BOOL "" FORCE)


set(GGML_NEON           ON  CACHE BOOL "" FORCE)
set(GGML_FP16_VA        OFF CACHE BOOL "" FORCE)
set(GGML_F16C           OFF CACHE BOOL "" FORCE)
set(GGML_DOTPROD        OFF CACHE BOOL "" FORCE)
set(GGML_I8MM           OFF CACHE BOOL "" FORCE)
set(GGML_SVE            OFF CACHE BOOL "" FORCE)


set(GGML_LLAMAFILE      OFF CACHE BOOL "" FORCE)


add_compile_definitions(USE_GREEDY_SAMPLER)

# ── BehaviorTree.cpp ──────────────────────────────────────────────────────────
set(BTCPP_SHARED_LIBS       OFF CACHE BOOL "" FORCE)
set(BTCPP_BUILD_TOOLS       OFF CACHE BOOL "" FORCE)
set(BTCPP_ENABLE_TESTING    OFF CACHE BOOL "" FORCE)
set(BTCPP_EXAMPLES          OFF CACHE BOOL "" FORCE)
set(BTCPP_GROOT_INTERFACE   OFF CACHE BOOL "" FORCE)
set(BTCPP_SQLITE_LOGGING    OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING           OFF CACHE BOOL "" FORCE)

add_subdirectory(llama.cpp)
add_subdirectory(behaviortree_cpp)

# ── Patch ggml targets ────────────────────────────────────────────────────────
foreach(target ggml-base ggml-cpu ggml llama)
    if(TARGET ${target})
        set_target_properties(${target} PROPERTIES LINKER_LANGUAGE CXX)
        target_link_libraries(${target} PRIVATE c++_shared)
        if(ANDROID_ABI STREQUAL "arm64-v8a")
            target_compile_options(${target} PRIVATE
                    -O3
                    -march=armv8-a
                    -fno-math-errno
                    -fno-trapping-math
                    -fno-finite-math-only
            )
        endif()
    endif()
endforeach()

# ── PDFium ────────────────────────────────────────────────────────────────────
set(PDFIUM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/pdfium/${ANDROID_ABI}")

add_library(pdfium SHARED IMPORTED)
set_target_properties(pdfium PROPERTIES
        IMPORTED_LOCATION "${PDFIUM_DIR}/lib/libpdfium.so"
)

# ── Main native library ───────────────────────────────────────────────────────
add_library(
        bt_agent_native SHARED
        native-lib.cpp
        llm_engine.cpp
        embedding_engine.cpp
        pdf_processor.cpp
        vector_store.cpp
        agent_bt.cpp
        tool_registry.cpp   # ← this is the only addition
)


target_compile_options(bt_agent_native PRIVATE
        -O3
        -fno-math-errno
        -fno-trapping-math
        -fno-finite-math-only
        -funroll-loops
        $<$<STREQUAL:${ANDROID_ABI},arm64-v8a>:-march=armv8-a>
)

target_include_directories(
        bt_agent_native PRIVATE
        llama.cpp/include
        llama.cpp/ggml/include
        behaviortree_cpp/include
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PDFIUM_DIR}/include
)

target_link_libraries(
        bt_agent_native
        llama
        ggml
        behaviortree_cpp
        pdfium
        log
        android
        c++_shared
)