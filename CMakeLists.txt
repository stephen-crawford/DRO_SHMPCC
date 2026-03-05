# OT-SHMPC Paper Export: Minimal CMake for paper experiments and figures
# Requires: Eigen3. Build: cmake -S . -B build && cmake --build build

cmake_minimum_required(VERSION 3.14)
project(ot_shmpc_paper VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(SCENARIO_MPC_SOURCES
    src/adaptive_dro_shift.cpp
    src/certificate_first.cpp
    src/collision_constraints.cpp
    src/conformal_safety.cpp
    src/contouring_mpc.cpp
    src/dynamics.cpp
    src/experiment_harness.cpp
    src/gan_scenario_loader.cpp
    src/hazard_switch_sampling.cpp
    src/mode_weights.cpp
    src/mpc_controller.cpp
    src/optimal_transport_predictor.cpp
    src/qp_solver.cpp
    src/reference_path.cpp
    src/risk_directed_bandit.cpp
    src/runtime_assurance.cpp
    src/scenario_compiler.cpp
    src/scenario_pruning.cpp
    src/scenario_sampler.cpp
    src/trajectory_moments.cpp
    src/dual_risk_monitor.cpp
    src/wasserstein_dro.cpp
)

add_library(scenario_mpc ${SCENARIO_MPC_SOURCES})
target_include_directories(scenario_mpc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(scenario_mpc PUBLIC Eigen3::Eigen)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(scenario_mpc PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

# Paper experiment runner (experiments A–AB, writes CSVs to paper_figures/)
add_executable(paper_experiment_runner tests/paper_experiment_runner.cpp)
target_link_libraries(paper_experiment_runner PRIVATE scenario_mpc)

# DRO / statistical power tests (extra CSVs for fig 10–12)
add_executable(test_dro_framework tests/test_dro_framework.cpp)
target_link_libraries(test_dro_framework PRIVATE scenario_mpc)

add_executable(test_statistical_power tests/test_statistical_power.cpp)
target_link_libraries(test_statistical_power PRIVATE scenario_mpc)
