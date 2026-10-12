#!/usr/bin/env python3
"""Execute reference certifier method bodies with ROS logging stubbed out."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

PRELUDE=r'''
#include <cmath>
#include <functional>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#define BISECTION_TOLERANCE 1e-6
#define LOG_INFO(x)
#define ROS_WARN_STREAM(x)
namespace RosTools {template<typename T>int sgn(T x){return (T(0)<x)-(x<T(0));}}
struct Config {bool enable_safe_horizon_=true;int removal_count_=0;} SCENARIO_CONFIG;
namespace ScenarioModule {
struct SafetyCertifier {
 double risk_,confidence_;int sample_size_,max_support_;
 int Bisection(double,double,std::function<double(int)>);
 int FindSampleSize(double,double,int);
 double Epsilon(int,int);
 double EpsilonForMaxSupport(int);
 double rootedNChooseK(double,double,double);
};
'''
POSTLUDE=r'''
}
int main(int argc,char** argv){
 if(argc!=3)return 2;
 std::ifstream in(argv[1]);std::ofstream out(argv[2]);std::string line;
 std::getline(in,line);out<<std::setprecision(17)<<"h,current_S,reference_S,reference_epsilon,target\n";
 while(std::getline(in,line)){
  std::stringstream s(line);std::string part;std::vector<std::string> f;
  while(std::getline(s,part,','))f.push_back(part);
  ScenarioModule::SafetyCertifier ref;ref.max_support_=std::stoi(f[0]);
  ref.risk_=std::stod(f[2]);ref.confidence_=std::stod(f[3]);
  int size=ref.FindSampleSize(ref.risk_,ref.confidence_,ref.max_support_);
  out<<ref.max_support_<<','<<f[1]<<','<<size<<','<<ref.Epsilon(size,ref.max_support_)<<','<<ref.risk_<<'\n';
 }
}
'''

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--reference',type=Path,required=True)
 ap.add_argument('--curve',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
 args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 path=args.reference/'src/safety_certifier.cpp';source=path.read_text();methods=[]
 for result,name in [('int','Bisection'),('int','FindSampleSize'),('double','Epsilon'),('double','EpsilonForMaxSupport'),('double','rootedNChooseK')]:
  start=source.index(f'  {result} SafetyCertifier::{name}(')
  end=source.index('\n  }',start)+len('\n  }')
  methods.append(source[start:end])
 generated=args.output/'extracted_reference_certifier.cpp'
 generated.write_text(PRELUDE+'\n'.join(methods)+POSTLUDE)
 with tempfile.TemporaryDirectory(prefix='shmpc-certifier-') as temp:
  binary=Path(temp)/'compare'
  subprocess.run(['g++','-std=c++17','-O2',str(generated),'-o',str(binary)],check=True)
  subprocess.run([str(binary),str(args.curve),str(args.output/'reference_sample_curve.csv')],check=True)
 (args.output/'certifier_source_hash.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2)+'\n')
 print((args.output/'reference_sample_curve.csv').read_text())

if __name__=='__main__':main()
