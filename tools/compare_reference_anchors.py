#!/usr/bin/env python3
"""Execute unmodified reference Push/DR method text on saved one-disc fixtures.

ROS/solver accessors are replaced by a position buffer; no optimizer is mocked
as successful. Only the straight one-obstacle geometry is compared.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile

PRELUDE = r'''
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <sys/types.h>
#define PROFILE_FUNCTION()
#define LOG_ERROR(x) std::cerr << x << '\n'
namespace RosTools { double distance(const Eigen::Vector2d& a,const Eigen::Vector2d& b){return (a-b).norm();} }
struct Config { int N_=20,max_obstacles_=1; } SCENARIO_CONFIG;
struct Disc {
 Eigen::Vector2d getPosition(Eigen::Vector2d p,double) const{return p;}
 Eigen::Vector2d toRobotCenter(Eigen::Vector2d p,double) const{return p;}
};
struct RealTimeData {std::vector<Disc> robot_area{Disc{}};};
struct Buffer {
 std::vector<Eigen::Vector2d> p=std::vector<Eigen::Vector2d>(21,Eigen::Vector2d::Zero());
 Eigen::Vector2d getEgoPredictionPosition(int k){return p.at(k);}
 double getEgoPrediction(int,const std::string&){return 0;}
 void setEgoPredictionPosition(int k,Eigen::Vector2d v){p.at(k)=v;}
};
namespace ScenarioModule {
struct SafeHorizon {
 int disc_id_=0,N=20,S;
 Buffer* _solver;
 std::vector<double> radii_{0.625};
 std::vector<std::vector<std::vector<Eigen::VectorXd>>>* scenarios_;
 void PushAlgorithm(const RealTimeData&);
 void DRProjection(const RealTimeData&);
};
'''
POSTLUDE = r'''
}
std::vector<std::string> split(const std::string& s){
 std::stringstream ss(s);std::string t;std::vector<std::string> out;
 while(std::getline(ss,t,','))out.push_back(t);return out;
}
int main(int argc,char** argv){
 if(argc!=5)return 2;
 Buffer buffer;std::ifstream anchors(argv[1]);std::string line;
 std::getline(anchors,line);
 while(std::getline(anchors,line)){
  auto f=split(line);int k=std::stoi(f.at(0));
  buffer.p.at(k)=Eigen::Vector2d(std::stod(f.at(4)),std::stod(f.at(5)));
 }
 struct Point {int id,k;double x,y;};std::vector<Point> points;int count=0;
 std::ifstream samples(argv[2]);std::getline(samples,line);
 while(std::getline(samples,line)){
  auto f=split(line);if(f.at(0)!=argv[3] || std::stoi(f.at(3))==0)continue;
  int id=std::stoi(f.at(1));count=std::max(count,id+1);
  points.push_back({id,std::stoi(f.at(3))-1,std::stod(f.at(4)),std::stod(f.at(5))});
 }
 std::vector<std::vector<std::vector<Eigen::VectorXd>>> batch(20,
  std::vector<std::vector<Eigen::VectorXd>>(1,std::vector<Eigen::VectorXd>(2,Eigen::VectorXd::Zero(count))));
 for(auto p:points){batch.at(p.k)[0][0](p.id)=p.x;batch.at(p.k)[0][1](p.id)=p.y;}
 ScenarioModule::SafeHorizon ref;ref.S=count;ref._solver=&buffer;ref.scenarios_=&batch;
 RealTimeData data;ref.PushAlgorithm(data);ref.DRProjection(data);
 std::ofstream out(argv[4]);out<<std::setprecision(17)<<"k,x,y\n";
 for(int k=1;k<=20;++k)out<<k<<','<<buffer.p[k].x()<<','<<buffer.p[k].y()<<'\n';
}
'''


def method(source, name, next_name):
    start=source.index('  void SafeHorizon::'+name+'(')
    end=source.index(next_name,start)
    return source[start:end]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--fixtures',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--dr-passes',type=int,default=5)
    ap.add_argument('--reflection',choices=['reference','radial'],default='reference')
    ap.add_argument('--force-passes',action='store_true')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    path=args.reference/'src/safe_horizon.cpp';text=path.read_text()
    dr=method(text,'DRProjection','  void SafeHorizon::PushAlgorithm(')
    push=method(text,'PushAlgorithm','  // Check for the x\'s')
    assert dr.count('int iterates = 5;') == 1
    dr=dr.replace('int iterates = 5;',f'int iterates = {args.dr_passes};')
    if args.reflection == 'radial':
        dr=dr.replace('delta - start_pose','delta - update_pose').replace('start_pose - delta','update_pose - delta')
    if args.force_passes:
        dr=dr.replace('if (RosTools::distance(prev_pose, pose) < 1e-5)',
                      'if (false && RosTools::distance(prev_pose, pose) < 1e-5)')
    generated=PRELUDE+dr+push+POSTLUDE
    (args.output/'extracted_reference_fixture.cpp').write_text(generated)
    summary=[]
    with tempfile.TemporaryDirectory(prefix='shmpc-reference-') as temporary:
        binary=Path(temporary)/'fixture'
        subprocess.run(['g++','-std=c++17','-O2','-I/usr/include/eigen3',
                        str(args.output/'extracted_reference_fixture.cpp'),'-o',str(binary)],check=True)
        for seed_dir in sorted(args.fixtures.glob('seed_*')):
            for batch,combo in [('gamma',1),('omega',2)]:
                out=args.output/f'{seed_dir.name}_{batch}.csv'
                with (args.output/f'{seed_dir.name}_{batch}.log').open('w') as log:
                    subprocess.run([str(binary),str(seed_dir/'rows_0.csv'),str(seed_dir/'samples.csv'),batch,str(out)],
                                   check=True,stdout=log,stderr=subprocess.STDOUT)
                anchors={int(r['k']):(float(r['x']),float(r['y'])) for r in csv.DictReader(out.open())}
                current={int(r['k']):(float(r['anchor_x']),float(r['anchor_y'])) for r in csv.DictReader((seed_dir/f'rows_{combo}.csv').open())}
                distances=[sum((anchors[k][j]-current[k][j])**2 for j in (0,1))**.5 for k in anchors]
                margins=[]
                for row in csv.DictReader((seed_dir/'samples.csv').open()):
                    k=int(row['k'])
                    if row['batch']!=batch or k==0: continue
                    x,y=anchors[k]
                    margins.append(math.hypot(x-float(row['x']),y-float(row['y']))-0.625)
                summary.append(dict(seed=seed_dir.name,batch=batch,passes=args.dr_passes,
                                    reflection=args.reflection,max_anchor_difference=max(distances),
                                    different_stages=sum(d>1e-8 for d in distances),
                                    min_clearance=min(margins),distance_violations=sum(d < -0.001 for d in margins)))
    (args.output/'reference_anchor_comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    (args.output/'reference_method_hashes.json').write_text(json.dumps(dict(
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        push_sha256=hashlib.sha256(push.encode()).hexdigest(),dr_sha256=hashlib.sha256(dr.encode()).hexdigest()),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
