#ifndef GLOBALS_H
#define GLOBALS_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <typeinfo>
#include <vector>
#include <despot/config.h>
#include <inttypes.h>

#include <despot/util/exec_tracker.h>
#include <despot/util/logging.h>

namespace despot {

typedef uint64_t OBS_TYPE;

namespace Globals {
extern const double NEG_INFTY;
extern const double POS_INFTY;
extern const double INF;
extern const double TINY;

extern Config config;
extern ExecTracker tracker;

inline bool Fequals(double a, double b)
{
  return std::fabs(a - b) < TINY;
}

inline double Discount()
{
  return config.discount;
}

inline double Discount(int d)
{
  return std::pow(config.discount, d);
}

inline void Track(std::string addr, std::string loc)
{
  tracker.Track(addr, loc);
}

inline void Untrack(std::string addr)
{
  tracker.Untrack(addr);
}

inline void PrintLocs()
{
  tracker.PrintLocs();
}
} // namespace

} // namespace despot

#endif
