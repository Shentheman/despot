#ifndef UPPER_BOUND_H
#define UPPER_BOUND_H

#include <cassert>
#include <vector>

#include <despot/core/history.h>
#include <despot/random_streams.h>

namespace despot {

class State;
class StateIndexer;
class DSPOMDP;
class Belief;
class MDP;
struct ValuedAction;

/* =============================================================================
 * ScenarioUpperBound class
 * =============================================================================*/

class ScenarioUpperBound
{
public:
  ScenarioUpperBound();
  virtual ~ScenarioUpperBound();

  virtual void Init(const RandomStreams& streams);

  virtual double Value(
      const std::vector<State*>& particles,
      RandomStreams& streams,
      History& history) const = 0;

  friend std::ostream& operator<<(
      std::ostream& os, const ScenarioUpperBound& belief);
  virtual std::string text() const;
};

/* =============================================================================
 * ParticleUpperBound class
 * =============================================================================*/

class ParticleUpperBound : public ScenarioUpperBound
{
public:
  ParticleUpperBound();
  virtual ~ParticleUpperBound();

  /**
   * Returns an upper bound to the maximum total discounted reward over an
   * infinite horizon for the (unweighted) particle.
   */
  virtual double Value(const State& state) const = 0;

  virtual double Value(
      const std::vector<State*>& particles,
      RandomStreams& streams,
      History& history) const;

  friend std::ostream& operator<<(
      std::ostream& os, const ParticleUpperBound& belief);
  virtual std::string text() const;
};

/* =============================================================================
 * TrivialParticleUpperBound class
 * =============================================================================*/

class TrivialParticleUpperBound : public ParticleUpperBound
{
protected:
  const DSPOMDP* model_;

public:
  TrivialParticleUpperBound(const DSPOMDP* model);
  virtual ~TrivialParticleUpperBound();

  double Value(const State& state) const;

  virtual double Value(
      const std::vector<State*>& particles,
      RandomStreams& streams,
      History& history) const;

  friend std::ostream& operator<<(
      std::ostream& os, const TrivialParticleUpperBound& belief);
  virtual std::string text() const;
};

/* =============================================================================
 * LookaheadUpperBound class
 * =============================================================================*/

class LookaheadUpperBound : public ScenarioUpperBound
{
protected:
  const DSPOMDP* model_;
  // Interface for a mapping between states and indices. In pomdp.h.
  const StateIndexer& indexer_;
  std::vector<std::vector<std::vector<double> > > bounds_;
  ParticleUpperBound* particle_upper_bound_;

public:
  LookaheadUpperBound(
      const DSPOMDP* model,
      const StateIndexer& indexer,
      ParticleUpperBound* bound);

  virtual void Init(const RandomStreams& streams);

  double Value(
      const std::vector<State*>& particles,
      RandomStreams& streams,
      History& history) const;

  friend std::ostream& operator<<(
      std::ostream& os, const LookaheadUpperBound& belief);
  virtual std::string text() const;
};

/* =============================================================================
 * BeliefUpperBound class
 * =============================================================================*/

class BeliefUpperBound
{
public:
  BeliefUpperBound();
  virtual ~BeliefUpperBound();

  virtual double Value(const Belief* belief) const = 0;

  friend std::ostream& operator<<(
      std::ostream& os, const BeliefUpperBound& belief);
  virtual std::string text() const;
};

/* =============================================================================
 * TrivialBeliefUpperBound class
 * =============================================================================*/

class TrivialBeliefUpperBound : public BeliefUpperBound
{
protected:
  const DSPOMDP* model_;

public:
  TrivialBeliefUpperBound(const DSPOMDP* model);

  double Value(const Belief* belief) const;

  friend std::ostream& operator<<(
      std::ostream& os, const TrivialBeliefUpperBound& belief);
  virtual std::string text() const;
};

/* =============================================================================
 * MDPUpperBound class
 * =============================================================================*/

class MDPUpperBound : public ParticleUpperBound, public BeliefUpperBound
{
protected:
  const MDP* model_;
  const StateIndexer& indexer_;
  std::vector<ValuedAction> policy_;

public:
  MDPUpperBound(const MDP* model, const StateIndexer& indexer);

  // shut off "hides overloaded virtual function" warning
  using ParticleUpperBound::Value;
  double Value(const State& state) const;

  double Value(const Belief* belief) const;

  friend std::ostream& operator<<(
      std::ostream& os, const MDPUpperBound& belief);
  virtual std::string text() const;
};

} // namespace despot

#endif
