#include "adversarial_bundles.hpp"
#include "collision_constraints.hpp"
#include "mpc_controller.hpp"
#include <iostream>
#include <set>
#include <stdexcept>
using namespace dro_mpc;
void require(bool value,const char* message) { if (!value) throw std::runtime_error(message); }
int main() {
    RuntimeConfig cfg;
    cfg.mpc.bundle_amplification=2;
    cfg.mpc.bundle_extra_draws=2;
    cfg.mpc.horizon=4;
    cfg.mpc.ego.num_discs=1;
    cfg.mpc.ego.length=0;
    cfg.random_seed=77;
    const ModeModel a("a",Eigen::Matrix4d::Identity(),Eigen::Vector4d::Zero(),Eigen::MatrixXd::Zero(4,2));
    const ModeModel b("b",Eigen::Matrix4d::Identity(),Eigen::Vector4d::Zero(),Eigen::MatrixXd::Zero(4,2));
    ModeHistory history(0,{{"a",a},{"b",b}});
    for(int i=0;i<100;++i)history.record_observation(i,0,i<90?"a":"b");
    const auto allocation=design_adversarial_bundles(history,{{"a",0},{"b",1}},{{"a",0},{"b",1}},cfg);
    require(allocation.multiplicities[0]>=1,"zero WDRO weight must retain the CP coverage floor");
    require(allocation.multiplicities[1]==static_cast<int>(std::ceil(2*allocation.upper[1]))+2,
        "WDRO extras must go to the mode with positive q times score");
    for(auto& [mode,model]:history.available_modes) {model.G(0,0)=.1;model.G(1,1)=.1;}
    std::mt19937 rng(77),rng_repeat(77);
    const auto noisy=sample_adversarial_bundles(0,ObstacleState(0,0,0,0),history,allocation,1,1000,rng);
    const auto noisy_repeat=sample_adversarial_bundles(0,ObstacleState(0,0,0,0),history,allocation,1,1000,rng_repeat);
    double mean=0,second_moment=0;
    for(std::size_t i=0;i<noisy.size();++i) {
        const auto x=noisy[i].trajectories.at(0).steps[1].mean;
        require(x==noisy_repeat[i].trajectories.at(0).steps[1].mean,"conditional draws must repeat exactly");
        mean+=x.x();second_moment+=x.x()*x.x();
    }
    mean/=noisy.size();second_moment/=noisy.size();
    require(std::abs(mean)<.01 && std::abs(second_moment-.01)<.002,"conditional Gaussian empirical moments");
    require(noisy[0].trajectories.at(0).steps[1].mean!=noisy[1].trajectories.at(0).steps[1].mean,
        "bundle must contain independent draws, not copied paths");
    MPCController controller(cfg);
    controller.initialize_obstacle(0,0,{{"a",a},{"b",b}});
    for (int i=0;i<100;++i) controller.update_mode_observation(0,0,i<90?"a":"b",i);
    // Run actual controller sampler, QP, support accounting and return guard.
    const auto result=controller.solve(EgoState(0,0,0,0),{{0,ObstacleState(5,3,0,0)}},{20,0});
    require(result.success && result.bundle_sampling,"bundle controller must return admissible fixture plan");
    require(result.sampled_scenarios==result.required_scenarios,"automatic sizing counts bundles");
    require(result.bundle_amplification>=2 && result.sampled_scenarios<cfg.compute_required_scenarios(),"coverage floor reduces macro sample count");
    require(result.raw_scenario_draws>result.sampled_scenarios,"raw draws must be logged separately");
    std::set<int> raw,groups;
    for (const auto& s:controller.scenarios()) {raw.insert(s.scenario_id);groups.insert(s.support_id());}
    require(raw.size()==controller.scenarios().size() && groups.size()==static_cast<std::size_t>(result.sampled_scenarios),"unique raw IDs and grouped support IDs");
    for (int id:result.support_scenarios) require(groups.count(id),"support IDs must refer to bundles");
    auto facets=compute_linearized_constraints(result.ego_trajectory,controller.scenarios(),.5,.35,.1,1,0);
    const auto kept=prune_bundle_constraints_exact(facets);
    require(kept.size()==groups.size()*4,"identical internal trajectories collapse to one facet per stage and bundle");
    require(scenario_group_count(controller.scenarios())==result.sampled_scenarios,"group count API");
    // A slightly different normal is NOT exact dominance; opposite normals and
    // identical facets from another bundle must also remain distinct.
    CollisionConstraint c(1,0,0,{1,0},1),lo(1,0,0,{1,0},.5),other(1,0,1,{1,0},1),tilt(1,0,0,{1,1e-14},.2);
    const auto pruned=prune_bundle_constraints_exact({c,lo,other,tilt});
    require(pruned.size()==3,"only exact within-bundle implication may prune");
    for (double x:{-2.,0.,.75,1.,3.}) for (double y:{-1e15,0.,1e15}) {
        bool before=true,after=true;
        for (const auto& row:std::vector<CollisionConstraint>{c,lo,other,tilt}) before &= row.evaluate({x,y})>=0;
        for (const auto& row:pruned) after &= row.evaluate({x,y})>=0;
        require(before==after,"pruning must preserve conjunction");
    }
    // Fresh controller with same seed reproduces every raw trajectory.
    MPCController repeat(cfg);repeat.initialize_obstacle(0,0,{{"a",a},{"b",b}});
    for (int i=0;i<100;++i) repeat.update_mode_observation(0,0,i<90?"a":"b",i);
    auto again=repeat.solve(EgoState(0,0,0,0),{{0,ObstacleState(5,3,0,0)}},{20,0});
    require(again.sampled_scenarios==result.sampled_scenarios && again.support_scenarios==result.support_scenarios,"repeat support/count");
    std::cout << "PASS bundle controller: baseline=" << cfg.compute_required_scenarios()
              << " bundles=" << result.sampled_scenarios << " raw=" << result.raw_scenario_draws
              << " c=" << result.bundle_amplification << " eta=" << result.bundle_threshold
              << " facets=" << kept.size() << " support=" << result.support_size << '\n';
    auto contact_cfg=cfg;
    contact_cfg.mpc.sampling.set_manual_sample_count(3);
    MPCController contact(contact_cfg);
    contact.initialize_obstacle(0,0,{{"a",a},{"b",b}});
    for(int i=0;i<100;++i)contact.update_mode_observation(0,0,i<90?"a":"b",i);
    const auto binding=contact.solve(EgoState(0,0,0,0),{{0,ObstacleState(.95,0,0,0)}},{10,0});
    require(binding.support_size==3,"multiple binding internal candidates must count exactly three bundles");
    require(binding.raw_scenario_draws>3 && !binding.sample_count_sufficient,
        "raw trajectories must not satisfy the macro-scenario sample requirement");
    std::cout << "PASS binding support: bundles=" << binding.sampled_scenarios << " raw=" << binding.raw_scenario_draws
              << " support=" << binding.support_size << '\n';
    cfg.mpc.type=MPCType::SH_MPCC_DRO_FALLBACK;
    cfg.validate(); // Approved extension allocates beta_cert across both attempts.
    cfg.mpc.bundle_extra_draws=0;
    cfg.mpc.bundle_amplification=2;
    MPCController multiple(cfg);
    // One mode requires no estimated categorical confidence interval.
    multiple.initialize_obstacle(0,0,{{"a",a}});
    multiple.initialize_obstacle(1,1,{{"a",a}});
    const auto multi=multiple.solve(EgoState(0,0,0,0),
        {{0,ObstacleState(5,3,0,0)},{1,ObstacleState(7,-3,0,0)}},{20,0});
    require(multi.success && multi.bundle_sampling,"multi-obstacle single-mode hybrid solve");
    const double expected_eta=-std::expm1(4*std::log1p(-cfg.epsilon()/2));
    require(std::abs(multi.bundle_threshold-expected_eta)<1e-15,"multi-obstacle union/AM-GM threshold");
    auto sizing=cfg;
    sizing.mpc.sampling.chance_of_certificate_violation/=2;
    sizing.mpc.sampling.one_minus_chance_constraint_violation_probability=1-expected_eta;
    require(multi.required_scenarios==sizing.compute_required_scenarios(),"hybrid allocates beta across two attempts");
    require(multi.raw_scenario_draws==4*multi.sampled_scenarios,"joint bundle includes candidates for both obstacles");
    require(multi.bundle_multiplicities.at("0:a")==2 && multi.bundle_multiplicities.at("1:a")==2,
        "obstacle-qualified allocation identifiers");
    ModeHistory empty(0,{{"a",a},{"b",b}});
    const auto cold=design_adversarial_bundles(empty,{{"a",.5},{"b",.5}},{},cfg);
    require(cold.upper==std::vector<double>({1,1}) && cold.lower==std::vector<double>({0,0}),
        "empty history must use the whole simplex, not invented observations");
    cfg.mpc.sampling.markov_jump_system=true;
    bool rejected=false;try{cfg.validate();}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"unsupported within-horizon switching must fail closed");
}
