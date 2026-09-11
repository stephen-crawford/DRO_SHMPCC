cmake_minimum_required(VERSION 3.14)
project(dro_shmpcc VERSION 1.0.0 LANGUAGES CXX)

include(CTest)

# Embed source provenance in every harness artifact.  A revision alone is not
# enough during local research work, so record whether CMake observed a clean
# tree when the executable was configured as well.
execute_process(
    COMMAND git rev-parse --verify HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RESULT_VARIABLE DRO_MPC_GIT_REVISION_RESULT
    OUTPUT_VARIABLE DRO_MPC_GIT_REVISION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if(NOT DRO_MPC_GIT_REVISION_RESULT EQUAL 0)
    set(DRO_MPC_GIT_REVISION "unknown")
endif()
execute_process(
    COMMAND git status --porcelain
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RESULT_VARIABLE DRO_MPC_SOURCE_TREE_RESULT
    OUTPUT_VARIABLE DRO_MPC_SOURCE_TREE_STATUS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if(DRO_MPC_SOURCE_TREE_RESULT EQUAL 0 AND DRO_MPC_SOURCE_TREE_STATUS STREQUAL "")
    set(DRO_MPC_SOURCE_TREE_STATE "clean")
elseif(DRO_MPC_SOURCE_TREE_RESULT EQUAL 0)
    set(DRO_MPC_SOURCE_TREE_STATE "dirty")
else()
    set(DRO_MPC_SOURCE_TREE_STATE "unknown")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# The controller uses acados's dense HPIPM QP interface.  Acados does not
# install a consistently named package on every platform, so resolve its
# include/lib layout explicitly and keep the root overrideable for CI.
set(ACADOS_ROOT "" CACHE PATH
    "acados installation root (contains include/ and lib/)")
set(_dro_mpc_acados_hints
    "${ACADOS_ROOT}"
    "$ENV{ACADOS_INSTALL_DIR}"
    "$ENV{ACADOS_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../acados")
find_path(DRO_MPC_ACADOS_INCLUDE_DIR
    NAMES acados_c/dense_qp_interface.h
    HINTS ${_dro_mpc_acados_hints}
    PATH_SUFFIXES include)
if(NOT DRO_MPC_ACADOS_INCLUDE_DIR)
    message(FATAL_ERROR
        "acados headers were not found. Set ACADOS_ROOT to an installation "
        "containing include/acados_c/dense_qp_interface.h.")
endif()
get_filename_component(DRO_MPC_ACADOS_ROOT "${DRO_MPC_ACADOS_INCLUDE_DIR}" DIRECTORY)
set(DRO_MPC_ACADOS_LIB_DIR "${DRO_MPC_ACADOS_ROOT}/lib")

find_library(DRO_MPC_ACADOS_LIBRARY NAMES acados
    HINTS "${DRO_MPC_ACADOS_LIB_DIR}" NO_DEFAULT_PATH)
find_library(DRO_MPC_HPIPM_LIBRARY NAMES hpipm
    HINTS "${DRO_MPC_ACADOS_LIB_DIR}" NO_DEFAULT_PATH)
find_library(DRO_MPC_BLASFEO_LIBRARY NAMES blasfeo
    HINTS "${DRO_MPC_ACADOS_LIB_DIR}" NO_DEFAULT_PATH)
find_library(DRO_MPC_QPOASES_LIBRARY NAMES qpOASES_e
    HINTS "${DRO_MPC_ACADOS_LIB_DIR}" NO_DEFAULT_PATH)
foreach(_dro_mpc_required_library
        DRO_MPC_ACADOS_LIBRARY
        DRO_MPC_HPIPM_LIBRARY
        DRO_MPC_BLASFEO_LIBRARY)
    if(NOT ${_dro_mpc_required_library})
        message(FATAL_ERROR
            "acados dependency ${_dro_mpc_required_library} was not found in "
            "${DRO_MPC_ACADOS_LIB_DIR}.")
    endif()
endforeach()

set(DRO_MPC_ACADOS_GIT_COMMIT "unknown")
if(EXISTS "${DRO_MPC_ACADOS_LIB_DIR}/git_commit_hash")
    file(READ "${DRO_MPC_ACADOS_LIB_DIR}/git_commit_hash" DRO_MPC_ACADOS_GIT_COMMIT)
    string(STRIP "${DRO_MPC_ACADOS_GIT_COMMIT}" DRO_MPC_ACADOS_GIT_COMMIT)
endif()

add_library(dro_mpc_acados INTERFACE)
target_include_directories(dro_mpc_acados INTERFACE
    "${DRO_MPC_ACADOS_INCLUDE_DIR}"
    "${DRO_MPC_ACADOS_INCLUDE_DIR}/hpipm/include"
    "${DRO_MPC_ACADOS_INCLUDE_DIR}/blasfeo/include")
target_link_libraries(dro_mpc_acados INTERFACE
    "${DRO_MPC_ACADOS_LIBRARY}"
    "${DRO_MPC_HPIPM_LIBRARY}"
    "${DRO_MPC_BLASFEO_LIBRARY}")
if(DRO_MPC_QPOASES_LIBRARY)
    target_link_libraries(dro_mpc_acados INTERFACE "${DRO_MPC_QPOASES_LIBRARY}")
endif()

# libacados itself has dynamic dependencies.  DT_RPATH rather than DT_RUNPATH
# makes this explicit root available to those indirect dependencies as well.
if(UNIX AND NOT APPLE)
    set(CMAKE_BUILD_RPATH "${DRO_MPC_ACADOS_LIB_DIR}")
    set(CMAKE_INSTALL_RPATH "${DRO_MPC_ACADOS_LIB_DIR}")
endif()

message(STATUS "Using acados HPIPM dense-QP backend: ${DRO_MPC_ACADOS_ROOT} (${DRO_MPC_ACADOS_GIT_COMMIT})")

set(SCENARIO_MPC_SOURCES
    src/collision_constraints.cpp
    src/dynamics.cpp
    src/environment_geometry.cpp
    src/experiment_artifacts.cpp
    src/experiment_harness.cpp
    src/mode_weights.cpp
    src/mpc_controller.cpp
    src/primal_ot.cpp
    src/qp_solver.cpp
    src/reference_path.cpp
    src/scenario_sampler.cpp
    src/dro.cpp
)

add_library(scenario_mpc ${SCENARIO_MPC_SOURCES})
target_include_directories(scenario_mpc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(scenario_mpc PUBLIC Eigen3::Eigen dro_mpc_acados)
target_compile_definitions(scenario_mpc PUBLIC
    DRO_MPC_DEFAULT_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/configs/default.yaml"
    DRO_MPC_ACADOS_GIT_COMMIT="${DRO_MPC_ACADOS_GIT_COMMIT}"
    DRO_MPC_GIT_REVISION="${DRO_MPC_GIT_REVISION}"
    DRO_MPC_SOURCE_TREE_STATE="${DRO_MPC_SOURCE_TREE_STATE}")

if(UNIX AND NOT APPLE)
    target_link_options(scenario_mpc INTERFACE
        "-Wl,-rpath,${DRO_MPC_ACADOS_LIB_DIR}"
        "-Wl,--disable-new-dtags")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(scenario_mpc PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

# Supported experiment entry point. It delegates every rollout to the canonical
# harness and writes a reproducibility/trace/visualization bundle by default.
add_executable(experiment_runner tools/experiment_runner.cpp)
target_link_libraries(experiment_runner PRIVATE scenario_mpc)

option(DRO_MPC_ENABLE_ROS2
    "Build the optional ROS 2/RViz artifact replayer (requires a sourced ROS 2 environment)"
    OFF)
if(DRO_MPC_ENABLE_ROS2)
    find_package(ament_cmake REQUIRED)
    find_package(rclcpp REQUIRED)
    find_package(geometry_msgs REQUIRED)
    find_package(nav_msgs REQUIRED)
    find_package(visualization_msgs REQUIRED)

    add_executable(dro_mpc_rviz_replay
        tools/rviz_replay.cpp
    )

    ament_target_dependencies(
        dro_mpc_rviz_replay
        rclcpp
        geometry_msgs
        nav_msgs
        visualization_msgs
    )

    install(
        TARGETS dro_mpc_rviz_replay
        DESTINATION lib/${PROJECT_NAME}
    )
endif()

add_executable(test_obstacle_class tests/test_obstacle_class.cpp)
target_link_libraries(test_obstacle_class PRIVATE scenario_mpc)

add_executable(test_collision_halfspaces tests/test_collision_halfspaces.cpp)
target_link_libraries(test_collision_halfspaces PRIVATE scenario_mpc)

add_executable(test_pruning_and_linearization tests/test_pruning_and_linearization.cpp)
target_link_libraries(test_pruning_and_linearization PRIVATE scenario_mpc)

add_executable(test_sample_complexity tests/test_sample_complexity.cpp)
target_link_libraries(test_sample_complexity PRIVATE scenario_mpc)

add_executable(test_dynamics tests/test_dynamics.cpp)
target_link_libraries(test_dynamics PRIVATE scenario_mpc)

add_executable(test_primal_ot tests/test_primal_ot.cpp)
target_link_libraries(test_primal_ot PRIVATE scenario_mpc)

add_executable(test_scenario_sampler tests/test_scenario_sampler.cpp)
target_link_libraries(test_scenario_sampler PRIVATE scenario_mpc)

add_executable(test_experiment_statistics tests/test_experiment_statistics.cpp)
target_link_libraries(test_experiment_statistics PRIVATE scenario_mpc)

add_executable(test_experiment_artifacts tests/test_experiment_artifacts.cpp)
target_link_libraries(test_experiment_artifacts PRIVATE scenario_mpc)
target_include_directories(test_experiment_artifacts PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tools)

add_executable(test_experiment_runner_cli tests/test_experiment_runner_cli.cpp)
add_dependencies(test_experiment_runner_cli experiment_runner)
target_compile_definitions(test_experiment_runner_cli PRIVATE
    EXPERIMENT_RUNNER_PATH="$<TARGET_FILE:experiment_runner>")

add_executable(test_safe_horizon_configuration tests/test_safe_horizon_configuration.cpp)
target_link_libraries(test_safe_horizon_configuration PRIVATE scenario_mpc)

add_executable(test_support_accounting tests/test_support_accounting.cpp)
target_link_libraries(test_support_accounting PRIVATE scenario_mpc)

add_executable(test_projection tests/test_projection.cpp)
target_link_libraries(test_projection PRIVATE scenario_mpc)

add_executable(test_mode_belief tests/test_mode_belief.cpp)
target_link_libraries(test_mode_belief PRIVATE scenario_mpc)

add_executable(test_mode_catalog tests/test_mode_catalog.cpp)
target_link_libraries(test_mode_catalog PRIVATE scenario_mpc)

add_executable(test_waypoint_trajectory tests/test_waypoint_trajectory.cpp)
target_link_libraries(test_waypoint_trajectory PRIVATE scenario_mpc)

add_executable(test_environment_topologies tests/test_environment_topologies.cpp)
target_link_libraries(test_environment_topologies PRIVATE scenario_mpc)

add_executable(test_switching_risk tests/test_switching_risk.cpp)
target_link_libraries(test_switching_risk PRIVATE scenario_mpc)

add_executable(test_risk_scoring_models tests/test_risk_scoring_models.cpp)
target_link_libraries(test_risk_scoring_models PRIVATE scenario_mpc)

add_executable(test_ground_cost tests/test_ground_cost.cpp)
target_link_libraries(test_ground_cost PRIVATE scenario_mpc)

add_executable(test_dro_ambiguity_sets tests/test_dro_ambiguity_sets.cpp)
target_link_libraries(test_dro_ambiguity_sets PRIVATE scenario_mpc)

add_executable(test_acados_qp_solver tests/test_acados_qp_solver.cpp)
target_link_libraries(test_acados_qp_solver PRIVATE scenario_mpc)

add_executable(test_acados_reproducibility tests/test_acados_reproducibility.cpp)
target_link_libraries(test_acados_reproducibility PRIVATE scenario_mpc)

add_executable(test_sh_anchor_preparation tests/test_sh_anchor_preparation.cpp)
target_link_libraries(test_sh_anchor_preparation PRIVATE scenario_mpc)

add_executable(test_bounded_braking tests/test_bounded_braking.cpp)
target_link_libraries(test_bounded_braking PRIVATE scenario_mpc)
# Test the private fallback directly without changing the controller interface.
target_compile_options(test_bounded_braking PRIVATE -fno-access-control)

add_executable(test_plan_acceptance tests/test_plan_acceptance.cpp)
target_link_libraries(test_plan_acceptance PRIVATE scenario_mpc)

add_executable(test_mpcc_progress tests/test_mpcc_progress.cpp)
target_link_libraries(test_mpcc_progress PRIVATE scenario_mpc)

add_executable(test_velocity_bounds tests/test_velocity_bounds.cpp)
target_link_libraries(test_velocity_bounds PRIVATE scenario_mpc)

add_executable(pipeline_verify tests/pipeline_verify.cpp)
target_link_libraries(pipeline_verify PRIVATE scenario_mpc)

if(BUILD_TESTING)
    function(register_scenario_mpc_test target label)
        add_test(NAME ${target} COMMAND $<TARGET_FILE:${target}>)
        set_tests_properties(${target} PROPERTIES
            LABELS "${label};deterministic"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    endfunction()

    # Focused tests cover every production component and run under CTest.
    register_scenario_mpc_test(test_collision_halfspaces unit)
    register_scenario_mpc_test(test_pruning_and_linearization unit)
    register_scenario_mpc_test(test_projection unit)
    register_scenario_mpc_test(test_dynamics unit)
    register_scenario_mpc_test(test_obstacle_class integration)
    register_scenario_mpc_test(test_mode_belief unit)
    register_scenario_mpc_test(test_mode_catalog unit)
    register_scenario_mpc_test(test_scenario_sampler unit)
    register_scenario_mpc_test(test_waypoint_trajectory unit)
    register_scenario_mpc_test(test_environment_topologies unit)
    register_scenario_mpc_test(test_primal_ot unit)
    register_scenario_mpc_test(test_ground_cost unit)
    register_scenario_mpc_test(test_dro_ambiguity_sets unit)
    register_scenario_mpc_test(test_switching_risk unit)
    register_scenario_mpc_test(test_risk_scoring_models unit)
    register_scenario_mpc_test(test_acados_qp_solver unit)
    register_scenario_mpc_test(test_velocity_bounds integration)
    register_scenario_mpc_test(test_mpcc_progress integration)
    register_scenario_mpc_test(test_sh_anchor_preparation unit)
    register_scenario_mpc_test(test_bounded_braking unit)
    register_scenario_mpc_test(test_plan_acceptance integration)
    register_scenario_mpc_test(test_experiment_statistics unit)
    register_scenario_mpc_test(test_experiment_artifacts integration)
    register_scenario_mpc_test(test_experiment_runner_cli integration)
    register_scenario_mpc_test(test_sample_complexity unit)
    register_scenario_mpc_test(test_safe_horizon_configuration unit)
    register_scenario_mpc_test(test_support_accounting integration)
    register_scenario_mpc_test(test_acados_reproducibility integration)
    register_scenario_mpc_test(pipeline_verify integration)

    # The optional ROS 2 target cannot be built in non-ROS environments, but a
    # ROS-enabled build still gets a command-line smoke test without needing RViz.
    if(DRO_MPC_ENABLE_ROS2)
        add_test(NAME test_rviz_replay_help COMMAND $<TARGET_FILE:dro_mpc_rviz_replay> --help)
        set_tests_properties(test_rviz_replay_help PROPERTIES
            LABELS "integration;ros2"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()
endif()

# Full-route base scenarios retain GIF and RViz bundles for user inspection.
if(BUILD_TESTING)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME test_scenario_preview_toggle
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_scenario_preview_toggle.py"
                $<TARGET_FILE:experiment_runner>)
    set_tests_properties(test_scenario_preview_toggle PROPERTIES LABELS "integration;deterministic")
    foreach(base_mode mpc mpcc sh_mpc sh_mpcc mpc_dynamic mpcc_dynamic sh_mpc_dynamic sh_mpcc_dynamic sh_mpc_dro_dynamic sh_mpcc_dro_dynamic sh_mpcc_no_noise_dynamic)
        foreach(obstacle_count 0 1)
            if(base_mode MATCHES "dynamic" AND obstacle_count EQUAL 0)
                continue()
            endif()
            set(base_case "${base_mode}_${obstacle_count}_obstacles")
            add_test(NAME "base_${base_case}"
                COMMAND "${Python3_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_base_scenario.py"
                    --runner $<TARGET_FILE:experiment_runner>
                    --config "${CMAKE_CURRENT_SOURCE_DIR}/configs/base_tests/${base_case}.yaml"
                    --output "${CMAKE_CURRENT_BINARY_DIR}/base-artifacts")
            set_tests_properties("base_${base_case}" PROPERTIES
                LABELS "base;integration;deterministic"
                TIMEOUT 1800)
        endforeach()
    endforeach()
endif()

if(DRO_MPC_ENABLE_ROS2)
    ament_package()
endif()
