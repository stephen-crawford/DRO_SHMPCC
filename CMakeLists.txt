# OT-SHMPC Paper Export: Minimal CMake for paper experiments and figures
# Requires: Eigen3. Build: cmake -S . -B build && cmake --build build

cmake_minimum_required(VERSION 3.14)
project(ot_shmpc_paper VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(SCENARIO_MPC_SOURCES
    src/collision_constraints.cpp
    src/dynamics.cpp
    src/experiment_harness.cpp
    src/mode_weights.cpp
    src/mpc_controller.cpp
    src/primal_ot.cpp
    src/qp_solver.cpp
    src/reference_path.cpp
    src/scenario_sampler.cpp
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

add_executable(test_obstacle_class tests/test_obstacle_class.cpp)
target_link_libraries(test_obstacle_class PRIVATE scenario_mpc)

add_executable(test_collision_halfspaces tests/test_collision_halfspaces.cpp)
target_link_libraries(test_collision_halfspaces PRIVATE scenario_mpc)

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
