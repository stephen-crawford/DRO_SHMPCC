#!/usr/bin/env python3
"""Execute the local reference sampler method against our sampler on one bank.

ROS/config accessors are stubbed. Only the reference shuffle seed is replaced,
so its otherwise entropy-seeded permutation can be paired with our replay seed.
The actual reference method is read at runtime, not copied into the repository.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile


ADAPTER = r'''
#include "scenario_sampler.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

#define SCENARIO_INFO(x) ((void)0)
#define ROS_WARN_ONCE(x) ((void)0)
#define PROFILE_FUNCTION() ((void)0)
#define ROSTOOLS_ASSERT(x, message) do { if (!(x)) throw std::runtime_error(message); } while (false)

using Mode = std::vector<dro_mpc::GaussianPredictionStep>;
struct Prediction { std::vector<Mode> modes; std::vector<double> probabilities; };
struct DynamicObstacle { Prediction prediction; };
using trajectory_sample = std::vector<std::vector<Eigen::VectorXd>>;
struct Config { int N_, sample_size_, batch_count_, max_obstacles_; } SCENARIO_CONFIG;

namespace RosTools {
Eigen::Matrix2d rotationMatrixFromHeading(double angle) {
    Eigen::Matrix2d result;
    result << std::cos(angle), std::sin(angle), -std::sin(angle), std::cos(angle);
    return result;
}
}
namespace ScenarioModule {
struct RandomGenerator {
    std::mt19937 engine;
    double Double() { return std::uniform_real_distribution<double>(0.0, 1.0)(engine); }
    int Int(int maximum) { return std::uniform_int_distribution<int>(0, maximum)(engine); }
};
class Sampler {
public:
    bool standard_samples_ready_ = true, samples_ready_ = false;
    RandomGenerator random_generator_;
    std::vector<std::vector<Eigen::Vector2d>> standard_samples_;
    std::vector<trajectory_sample> samples_;
    std::vector<std::vector<std::vector<Eigen::Matrix2d>>> R_, SVD_, Sigma_, A_;
    std::vector<std::vector<double>> sum_of_probabilities_;
    int SampleSize() const { return standard_samples_.front().size(); }
    std::vector<trajectory_sample>* IntegrateAndTranslateToMeanAndVariance(
        const std::vector<DynamicObstacle>&, const double);
};
}
'''

FIXTURE = r'''
namespace dro_mpc {
struct DirectTrajectorySamplerTestAccess {
    static auto& bank(DirectTrajectorySampler& sampler) { return sampler.standard_samples_; }
};
}
int main() {
    constexpr int N=5, S=127, B=3, V=2;
    constexpr double dt=0.17;
    SCENARIO_CONFIG={N,S,B,V};
    dro_mpc::DirectTrajectorySampler candidate;
    ScenarioModule::Sampler reference;
    std::mt19937 bank_rng(104), candidate_rng(77);
    std::normal_distribution<double> normal;
    std::vector<std::vector<Eigen::Vector2d>> bank(B, std::vector<Eigen::Vector2d>(S));
    for (auto& batch:bank) for (auto& z:batch) { z.x()=normal(bank_rng); z.y()=normal(bank_rng); }
    dro_mpc::DirectTrajectorySamplerTestAccess::bank(candidate)=bank;
    reference.standard_samples_=bank;
    reference.random_generator_.engine=candidate_rng;
    reference.samples_.resize(N, trajectory_sample(V, std::vector<Eigen::VectorXd>(2, Eigen::VectorXd::Zero(S))));
    const auto matrices=std::vector<std::vector<std::vector<Eigen::Matrix2d>>>(
        V,std::vector<std::vector<Eigen::Matrix2d>>(N,std::vector<Eigen::Matrix2d>(2)));
    reference.R_=reference.SVD_=reference.Sigma_=reference.A_=matrices;
    reference.sum_of_probabilities_.resize(V);
    std::map<int,dro_mpc::ObstacleState> states;
    dro_mpc::ObstaclePredictions predictions;
    std::vector<DynamicObstacle> reference_obstacles(V);
    for (int v=0;v<V;++v) {
        const int id=3+v*8;
        states.emplace(id,dro_mpc::ObstacleState(v,2*v,0,0));
        for (int m=0;m<2;++m) {
            dro_mpc::GaussianPredictionMode mode;
            mode.mode_id=m==0?"first":"second";
            mode.probability=m==0?0.35:0.65;
            for (int k=0;k<N;++k)
                mode.steps.push_back({Eigen::Vector2d(10*m+v+0.3*k,-5*m+v-0.2*k),
                                      0.41+v*0.2+k*0.15, 0.7+0.1*k+0.2*m, 0.2+0.05*v});
            reference_obstacles[v].prediction.modes.push_back(mode.steps);
            reference_obstacles[v].prediction.probabilities.push_back(mode.probability);
            predictions[id].push_back(mode);
        }
    }
    double maximum_error=0.0;
    for (int call=0;call<2;++call) {
        const auto actual=candidate.sample(states,predictions,N,S,dt,B,candidate_rng,40);
        const auto expected=reference.IntegrateAndTranslateToMeanAndVariance(reference_obstacles,dt);
        for (int k=0;k<N;++k) for (int v=0;v<V;++v) for (int s=0;s<S;++s) {
            const Eigen::Vector2d target((*expected)[k][v][0](s),(*expected)[k][v][1](s));
            const auto& step=actual[s].trajectories.at(3+v*8).steps[k+1];
            maximum_error=std::max(maximum_error,(step.mean-target).norm());
        }
        if (candidate_rng!=reference.random_generator_.engine) {
            std::cerr<<"reference and candidate consumed different RNG draws\n";
            return 1;
        }
    }
    std::cout<<"Reference comparison: 2 calls, N=5, S=127, obstacles=2, modes=2, batches=3, dt=0.17\n"
             <<"Compared 2540 future positions; maximum position error="<<maximum_error<<"\n"
             <<"Reference batch/mode RNG state equals candidate RNG state after both calls\n";
    return maximum_error<1e-12?0:1;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=Path('/home/stephen/ros2_ws/src/scenario_module/src/sampler.cpp'))
    parser.add_argument('--cxx', default='c++')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    source = args.reference.read_text()
    print('Reference:', args.reference, 'sha256='+hashlib.sha256(source.encode()).hexdigest(), flush=True)
    start = source.index('  std::vector<trajectory_sample> *Sampler::IntegrateAndTranslateToMeanAndVariance(')
    end = source.index('  void Sampler::SamplePartitionedScenarios(', start)
    method = source[start:end]
    entropy = 'auto rng = std::mt19937{std::random_device{}()};'
    if method.count(entropy) != 1:
        raise RuntimeError('reference shuffle initialization changed; inspect the adapter')
    method = method.replace(entropy, 'auto rng = std::mt19937{random_generator_.engine()};')
    with tempfile.TemporaryDirectory(prefix='reference-sampling-') as directory:
        fixture = Path(directory) / 'comparison.cpp'
        fixture.write_text(ADAPTER + '\nnamespace ScenarioModule {\n' + method + '\n}\n' + FIXTURE)
        executable = Path(directory) / 'comparison'
        subprocess.run([args.cxx, '-std=c++17', '-O2', '-I'+str(repo/'include'), '-I/usr/include/eigen3',
                        str(fixture), str(repo/'src/direct_trajectory_sampler.cpp'), '-o', str(executable)], check=True)
        subprocess.run([str(executable)], check=True)


if __name__ == '__main__':
    main()
