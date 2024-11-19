set(CMAKE_CXX_STANDARD 11)

add_executable(test_seed_points test_seed_points.cpp)
target_link_libraries(test_seed_points ${PROJECT_NAME} -ltcmalloc)

# add_executable(test_nsg_search test_nsg_search.cpp)
# target_link_libraries(test_nsg_search ${PROJECT_NAME})

add_executable(test_searching test_searching.cpp)
target_link_libraries(test_searching ${PROJECT_NAME} -ltcmalloc)

