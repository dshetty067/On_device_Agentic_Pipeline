cmake_minimum_required(VERSION 3.22.1)
project(bt_agent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_ANDROID_STL_TYPE c++_shared)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")

set(LLAMA_STATIC ON)
set(GGML_METAL  OFF)
set(GGML_CUDA   OFF)
set(GGML_OPENMP OFF)

# ── BehaviorTree.cpp ───────────────────────────────────────────────────────
set(BTCPP_SHARED_LIBS       OFF CACHE BOOL "" FORCE)
set(BTCPP_BUILD_TOOLS       OFF CACHE BOOL "" FORCE)
set(BTCPP_ENABLE_TESTING    OFF CACHE BOOL "" FORCE)
set(BTCPP_EXAMPLES          OFF CACHE BOOL "" FORCE)
set(BTCPP_GROOT_INTERFACE   OFF CACHE BOOL "" FORCE)
set(BTCPP_SQLITE_LOGGING    OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING           OFF CACHE BOOL "" FORCE)

add_subdirectory(llama.cpp)
add_subdirectory(behaviortree_cpp)

# Patch ggml targets
foreach(target ggml-base ggml-cpu ggml llama)
    if(TARGET ${target})
        set_target_properties(${target} PROPERTIES LINKER_LANGUAGE CXX)
        target_link_libraries(${target} PRIVATE c++_shared)
    endif()
endforeach()

add_library(
        bt_agent_native SHARED
        native-lib.cpp
        llm_engine.cpp
        embedding_engine.cpp
        pdf_processor.cpp
        vector_store.cpp
        agent_bt.cpp
)

target_include_directories(
        bt_agent_native PRIVATE
        llama.cpp/include
        llama.cpp/ggml/include
        behaviortree_cpp/include
        ${CMAKE_CURRENT_SOURCE_DIR}   # ← picks up hnswlib/hnswlib.h (header only)
)

target_link_libraries(
        bt_agent_native
        llama
        ggml
        behaviortree_cpp
        log
        android
        c++_shared
)