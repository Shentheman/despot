#include <despot/core/lower_bound.h>
#include <despot/core/pomdp.h>
#include <despot/core/node.h>
#include <despot/solver/pomcp.h>

using namespace std;

namespace despot {

/* =============================================================================
 * ValuedAction class
 * =============================================================================*/

ValuedAction::ValuedAction() :
	action(-1),
	value(0) {
}

ValuedAction::ValuedAction(int _action, double _value) :
	action(_action),
	value(_value) {
}

ostream& operator<<(ostream& os, const ValuedAction& va) {
	os << "(" << va.action << ", " << va.value << ")";
	return os;
}

/* =============================================================================
 * ScenarioLowerBound class
 * =============================================================================*/

ScenarioLowerBound::ScenarioLowerBound(const DSPOMDP* model, Belief* belief) :
	Solver(model, belief) {
}

void ScenarioLowerBound::Init(const RandomStreams& streams) {
  cout << "[ScenarioLowerBound::Init()]" << endl;
}

void ScenarioLowerBound::Reset() {
}

ValuedAction ScenarioLowerBound::Search() {
	RandomStreams streams(Globals::config.num_scenarios,
		Globals::config.search_depth);
	vector<State*> particles = belief_->Sample(Globals::config.num_scenarios);

	ValuedAction va = Value(particles, streams, history_);

	for (int i = 0; i < particles.size(); i++)
		model_->Free(particles[i]);

	return va;
}

void ScenarioLowerBound::Learn(VNode* tree) {
}


string ScenarioLowerBound::text() const {
	return "ScenarioLowerBound";
}

ostream& operator<<(ostream& os, const ScenarioLowerBound& b) {
	os << (&b)->text();
	return os;
}


/* =============================================================================
 * POMCPScenarioLowerBound class
 * =============================================================================*/

POMCPScenarioLowerBound::POMCPScenarioLowerBound(const DSPOMDP* model,
	POMCPPrior* prior,
	Belief* belief) :
	ScenarioLowerBound(model, belief),
	prior_(prior) {
	explore_constant_ = model_->GetMaxReward()
		- model_->GetMinRewardAction().value;
}

ValuedAction POMCPScenarioLowerBound::Value(const vector<State*>& particles,
	RandomStreams& streams, History& history) const {
	prior_->history(history);
	VNode* root = POMCP::CreateVNode(0, particles[0], prior_, model_);
	// Note that particles are assumed to be of equal weight
	for (int i = 0; i < particles.size(); i++) {
		State* particle = particles[i];
		State* copy = model_->Copy(particle);
		POMCP::Simulate(copy, streams, root, model_, prior_);
		model_->Free(copy);
	}

	ValuedAction va = POMCP::OptimalAction(root);
	va.value *= State::Weight(particles);
	delete root;
	return va;
}


string POMCPScenarioLowerBound::text() const {
	return "POMCPScenarioLowerBound";
}

ostream& operator<<(ostream& os, const POMCPScenarioLowerBound& b) {
	os << (&b)->text();
	return os;
}


/* =============================================================================
 * ParticleLowerBound class
 * =============================================================================*/

ParticleLowerBound::ParticleLowerBound(const DSPOMDP* model, Belief* belief) :
	ScenarioLowerBound(model, belief) {
}

ValuedAction ParticleLowerBound::Value(const vector<State*>& particles,
	RandomStreams& streams, History& history) const {
	return Value(particles);
}


string ParticleLowerBound::text() const {
	return "ParticleLowerBound";
}

ostream& operator<<(ostream& os, const ParticleLowerBound& b) {
	os << (&b)->text();
	return os;
}


/* =============================================================================
 * TrivialParticleLowerBound class
 * =============================================================================*/

TrivialParticleLowerBound::TrivialParticleLowerBound(const DSPOMDP* model) :
	ParticleLowerBound(model) {
}

ValuedAction TrivialParticleLowerBound::Value(
	const vector<State*>& particles) const {
  // Similar to 4.4 in the journal paper, we can develop a fixed-action policy.
  // In this policy, we keep executing the action with the most immediate 
  // reward.
  // So the worst case is that we always get the min_reward of that action.
  // va = min_reward + min_reward * Discount + min_reward * Discount^2 ...
  //    = min_reward * (1-Discount^infty) / (1-Discount)
  //    = min_reward / (1-Discount)
  //
  // The min_reward is the worst reward of the best action.
  // So min_reward = Among the worst cases of all the actions,
  // the best action with its reward value.
	ValuedAction va = model_->GetMinRewardAction();
  // We also multiply the total weight in particles to va.
	va.value *= State::Weight(particles) / (1 - Globals::Discount());
	return va;
}


string TrivialParticleLowerBound::text() const {
	return "TrivialParticleLowerBound";
}

ostream& operator<<(ostream& os, const TrivialParticleLowerBound& b) {
	os << (&b)->text();
	return os;
}


/* =============================================================================
 * BeliefLowerBound class
 * =============================================================================*/

BeliefLowerBound::BeliefLowerBound(const DSPOMDP* model, Belief* belief) :
	Solver(model, belief) {
}

ValuedAction BeliefLowerBound::Search() {
	return Value(belief_);
}

void BeliefLowerBound::Learn(VNode* tree) {
}


string BeliefLowerBound::text() const {
	return "BeliefLowerBound";
}

ostream& operator<<(ostream& os, const BeliefLowerBound& b) {
	os << (&b)->text();
	return os;
}


/* =============================================================================
 * TrivialBeliefLowerBound class
 * =============================================================================*/

TrivialBeliefLowerBound::TrivialBeliefLowerBound(const DSPOMDP* model,
	Belief* belief) :
	BeliefLowerBound(model, belief) {
}

ValuedAction TrivialBeliefLowerBound::Value(const Belief* belief) const {
	ValuedAction va = model_->GetMinRewardAction();
	va.value *= 1.0 / (1 - Globals::Discount());
	return va;
}


string TrivialBeliefLowerBound::text() const {
	return "TrivialBeliefLowerBound";
}

ostream& operator<<(ostream& os, const TrivialBeliefLowerBound& b) {
	os << (&b)->text();
	return os;
}


} // namespace despot
