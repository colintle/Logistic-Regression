cmake_minimum_required(VERSION 3.14)
project(LogisticRegression)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/b2b9072ecbe874f5937054653ef8f2731eb0f010.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_subdirectory(matrix-library)

add_library(LogisticRegression STATIC logistic_regression.cc)
target_link_libraries(LogisticRegression PUBLIC MatrixLibraryAPI)
target_include_directories(LogisticRegression PUBLIC ${CMAKE_SOURCE_DIR})

# Testing
include(GoogleTest)

add_executable(LogisticRegressionTests logistic_regression_test.cc)
# Link tests against the LogisticRegression library so the symbols resolve
target_link_libraries(LogisticRegressionTests PRIVATE LogisticRegression gtest_main)
gtest_discover_tests(LogisticRegressionTests)