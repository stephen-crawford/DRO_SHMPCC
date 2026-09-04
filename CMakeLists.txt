# OT-SHMPC Paper Export: Minimal CMake for paper experiments and figures
# Requires: Eigen3. Build: cmake -S . -B build && cmake --build build

cmake_minimum_required(VERSION 3.14)
project(ot_shmpc_paper VERSION 1.0.0 LANGUAGES CXX)

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
    DRO_MPC_ACADOS_GIT_COMMIT="${DRO_MPC_ACADOS_GIT_COMMIT}")

if(UNIX AND NOT APPLE)
    target_link_options(scenario_mpc INTERFACE
        "-Wl,-rpath,${DRO_MPC_ACADOS_LIB_DIR}"
        "-Wl,--disable-new-dtags")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(scenario_mpc PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

# Paper experiment runner (experiments A–AB, writes CSVs to paper_figures/)
add_executable(paper_experiment_runner tests/paper_experiment_runner.cpp)
target_link_libraries(paper_experiment_runner PRIVATE scenario_mpc)

# DRO / statistical power tests (extra CSVs for fig 10–12)
add_executable(test_dro_framework tests/test_dro_framework.cpp)
target_link_libraries(test_dro_framework PRIVATE scenario_mpc)

add_executable(test_obstacle_class tests/test_obstacle_class.cpp)
target_link_libraries(test_obstacle_class PRIVATE scenario_mpc)

add_executable(test_collision_halfspaces tests/test_collision_halfspaces.cpp)
target_link_libraries(test_collision_halfspaces PRIVATE scenario_mpc)

add_executable(test_pruning_and_linearization tests/test_pruning_and_linearization.cpp)
target_link_libraries(test_pruning_and_linearization PRIVATE scenario_mpc)

add_executable(test_sample_complexity tests/test_sample_complexity.cpp)
target_link_libraries(test_sample_complexity PRIVATE scenario_mpc)

add_executable(test_projection tests/test_projection.cpp)
target_link_libraries(test_projection PRIVATE scenario_mpc)

add_executable(test_paper_strategies tests/test_paper_strategies.cpp)
target_link_libraries(test_paper_strategies PRIVATE scenario_mpc)

add_executable(test_strategy_sweep tests/test_strategy_sweep.cpp)
target_link_libraries(test_strategy_sweep PRIVATE scenario_mpc)

add_executable(test_tuning_sweep tests/test_tuning_sweep.cpp)
target_link_libraries(test_tuning_sweep PRIVATE scenario_mpc)

add_executable(test_paper_figures tests/test_paper_figures.cpp)
target_link_libraries(test_paper_figures PRIVATE scenario_mpc)

add_executable(test_dro_benefits tests/test_dro_benefits.cpp)
target_link_libraries(test_dro_benefits PRIVATE scenario_mpc)

add_executable(test_cdc_experiments tests/test_cdc_experiments.cpp)
target_link_libraries(test_cdc_experiments PRIVATE scenario_mpc)

add_executable(test_path_obstacle_sweep tests/test_path_obstacle_sweep.cpp)
target_link_libraries(test_path_obstacle_sweep PRIVATE scenario_mpc)

add_executable(test_generalization tests/test_generalization.cpp)
target_link_libraries(test_generalization PRIVATE scenario_mpc)

add_executable(test_trajectory_dro tests/test_trajectory_dro.cpp)
target_link_libraries(test_trajectory_dro PRIVATE scenario_mpc)

add_executable(radius_probe tests/radius_probe.cpp)
target_link_libraries(radius_probe PRIVATE scenario_mpc)

add_executable(micro_dro_probe tests/micro_dro_probe.cpp)
target_link_libraries(micro_dro_probe PRIVATE scenario_mpc)

add_executable(test_mode_belief tests/test_mode_belief.cpp)
target_link_libraries(test_mode_belief PRIVATE scenario_mpc)

add_executable(test_mode_catalog tests/test_mode_catalog.cpp)
target_link_libraries(test_mode_catalog PRIVATE scenario_mpc)

add_executable(test_environment_topologies tests/test_environment_topologies.cpp)
target_link_libraries(test_environment_topologies PRIVATE scenario_mpc)

add_executable(test_switching_risk tests/test_switching_risk.cpp)
target_link_libraries(test_switching_risk PRIVATE scenario_mpc)

add_executable(test_ground_cost tests/test_ground_cost.cpp)
target_link_libraries(test_ground_cost PRIVATE scenario_mpc)

add_executable(test_dro_ambiguity_sets tests/test_dro_ambiguity_sets.cpp)
target_link_libraries(test_dro_ambiguity_sets PRIVATE scenario_mpc)

add_executable(test_acados_qp_solver tests/test_acados_qp_solver.cpp)
target_link_libraries(test_acados_qp_solver PRIVATE scenario_mpc)

add_executable(test_acados_reproducibility tests/test_acados_reproducibility.cpp)
target_link_libraries(test_acados_reproducibility PRIVATE scenario_mpc)

add_executable(velocity_bound_probe tests/velocity_bound_probe.cpp)
target_link_libraries(velocity_bound_probe PRIVATE scenario_mpc)

add_executable(paper_recompute tests/paper_recompute.cpp)
target_link_libraries(paper_recompute PRIVATE scenario_mpc)

add_executable(paper_recompute_coverage tests/paper_recompute_coverage.cpp)
target_link_libraries(paper_recompute_coverage PRIVATE scenario_mpc)

add_executable(diagnose_wdro_benefit tests/diagnose_wdro_benefit.cpp)
target_link_libraries(diagnose_wdro_benefit PRIVATE scenario_mpc)

add_executable(shift_test tests/shift_test.cpp)
target_link_libraries(shift_test PRIVATE scenario_mpc)

add_executable(wdro_mechanism_probe tests/wdro_mechanism_probe.cpp)
target_link_libraries(wdro_mechanism_probe PRIVATE scenario_mpc)

add_executable(fixed_rho_test tests/fixed_rho_test.cpp)
target_link_libraries(fixed_rho_test PRIVATE scenario_mpc)

add_executable(injection_test tests/injection_test.cpp)
target_link_libraries(injection_test PRIVATE scenario_mpc)

add_executable(pipeline_verify tests/pipeline_verify.cpp)
target_link_libraries(pipeline_verify PRIVATE scenario_mpc)

add_executable(risk_and_rho_verify tests/risk_and_rho_verify.cpp)
target_link_libraries(risk_and_rho_verify PRIVATE scenario_mpc)

add_executable(ablation_confound_check tests/ablation_confound_check.cpp)
target_link_libraries(ablation_confound_check PRIVATE scenario_mpc)
