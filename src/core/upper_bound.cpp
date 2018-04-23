#include <despot/core/upper_bound.h>
#include <despot/core/pomdp.h>
#include <despot/core/mdp.h>

using namespace std;

namespace despot {

/* =============================================================================
 * ScenarioUpperBound
 * =============================================================================*/

ScenarioUpperBound::ScenarioUpperBound() {
}

ScenarioUpperBound::~ScenarioUpperBound() {
}

void ScenarioUpperBound::Init(const RandomStreams& streams) {
  if (logging::level() >= logging::DEBUG)
  {
  ROS_WARN_STREAM("[ScenarioUpperBound::Init]");
  }
}

string ScenarioUpperBound::text() const {
	return "ScenarioUpperBound";
}

ostream& operator<<(ostream& os, const ScenarioUpperBound& b) {
	os << (&b)->text();
	return os;
}



/* =============================================================================
 * ParticleUpperBound
 * =============================================================================*/

ParticleUpperBound::ParticleUpperBound() {
}

ParticleUpperBound::~ParticleUpperBound() {
}

double ParticleUpperBound::Value(const vector<State*>& particles,
	RandomStreams& streams, History& history) const {
	double value = 0;
	for (int i = 0; i < particles.size(); i++) {
		State* particle = particles[i];
		value += particle->weight * Value(*particle);
	}
	return value;
}

string ParticleUpperBound::text() const {
	return "ParticleUpperBound";
}

ostream& operator<<(ostream& os, const ParticleUpperBound& b) {
	os << (&b)->text();
	return os;
}


/* =============================================================================
 * TrivialParticleUpperBound
 * =============================================================================*/

TrivialParticleUpperBound::TrivialParticleUpperBound(const DSPOMDP* model) :
	model_(model) {
}

TrivialParticleUpperBound::~TrivialParticleUpperBound() {
}

double TrivialParticleUpperBound::Value(const State& state) const {
  // Uninformed bound
  // Assume the policy can always execute the actions with the highest
  // rewards, so val = max_reward + max_reward * Discount
  // + max_reward * Discount^2 + ...
  // = max_reward * (1-Discount^infty) / (1-Discount)
  // = max_reward / (1-Discount)
	return model_->GetMaxReward() / (1 - Globals::Discount());
}

double TrivialParticleUpperBound::Value(const vector<State*>& particles,
	RandomStreams& streams, History& history) const {
	return State::Weight(particles) * model_->GetMaxReward() / (1 - Globals::Discount());
}

string TrivialParticleUpperBound::text() const {
	return "TrivialParticleUpperBound";
}

ostream& operator<<(ostream& os, const TrivialParticleUpperBound& b) {
	os << (&b)->text();
	return os;
}


/* =============================================================================
 * LookaheadUpperBound
 * =============================================================================*/

LookaheadUpperBound::LookaheadUpperBound(const DSPOMDP* model,
	const StateIndexer& indexer, ParticleUpperBound* bound) :
	model_(model),
	indexer_(indexer),
	particle_upper_bound_(bound) {
}

void LookaheadUpperBound::Init(const RandomStreams& streams) {
  ROS_WARN_STREAM("[LookaheadUpperBound::Init]");

	int num_states = indexer_.NumStates();
	int length = streams.Length();
	int num_particles = streams.NumStreams();
  cout << "[LookaheadUpperBound::Init()]" << endl;

  cout << "num_states="<< num_states<<", length="<<length
    <<", num_particles="<<num_particles<<endl;

	SetSize(bounds_, num_particles, length + 1, num_states);

	clock_t start = clock();
	for (int p = 0; p < num_particles; p++) {
		if (p % 10 == 0)
			cerr << p << " scenarios done! ["
				<< (double(clock() - start) / CLOCKS_PER_SEC) << "s]" << endl;
		for (int t = length; t >= 0; t--) {
			if (t == length) { // base case
				for (int s = 0; s < num_states; s++) {
					bounds_[p][t][s] = particle_upper_bound_->Value(*indexer_.GetState(s));
				}
			} else { // lookahead
				for (int s = 0; s < num_states; s++) {
					double best = Globals::NEG_INFTY;

					for (int a = 0; a < model_->NumActions(); a++) {
						double reward = 0;
						State* copy = model_->Copy(indexer_.GetState(s));
						bool terminal = model_->Step(*copy, streams.Entry(p, t),
							a, reward);
						model_->Free(copy);
						reward += (!terminal) * Globals::Discount()
							* bounds_[p][t + 1][indexer_.GetIndex(copy)];

						if (reward > best)
							best = reward;
					}

					bounds_[p][t][s] = best;
				}
			}
		}
	}
}

double LookaheadUpperBound::Value(const vector<State*>& particles,
	RandomStreams& streams, History& history) const {
	double bound = 0;
	for (int i = 0; i < particles.size(); i++) {
		State* particle = particles[i];
		bound +=
			particle->weight
				* bounds_[particle->scenario_id][streams.position()][indexer_.GetIndex(
					particle)];
	}
	return bound;
}


string LookaheadUpperBound::text() const {
	return "LookaheadUpperBound";
}

ostream& operator<<(ostream& os, const LookaheadUpperBound& b) {
	os << (&b)->text();
	return os;
}




/* =============================================================================
 * BeliefUpperBound
 * =============================================================================*/

BeliefUpperBound::BeliefUpperBound() {
}

BeliefUpperBound::~BeliefUpperBound() {
}

string BeliefUpperBound::text() const {
	return "BeliefUpperBound";
}

ostream& operator<<(ostream& os, const BeliefUpperBound& b) {
	os << (&b)->text();
	return os;
}

TrivialBeliefUpperBound::TrivialBeliefUpperBound(const DSPOMDP* model) :
	model_(model) {
}

double TrivialBeliefUpperBound::Value(const Belief* belief) const {
	return model_->GetMaxReward() / (1 - Globals::Discount());
}

string TrivialBeliefUpperBound::text() const {
	return "TrivialBeliefUpperBound";
}

ostream& operator<<(ostream& os, const TrivialBeliefUpperBound& b) {
	os << (&b)->text();
	return os;
}








/* =============================================================================
 * MDPUpperBound
 * =============================================================================*/

MDPUpperBound::MDPUpperBound(const MDP* model,
	const StateIndexer& indexer) :
	model_(model),
	indexer_(indexer) {
	const_cast<MDP*>(model_)->ComputeOptimalPolicyUsingVI();
	policy_ = model_->policy();
}

double MDPUpperBound::Value(const State& state) const {
	return policy_[indexer_.GetIndex(&state)].value;
}

double MDPUpperBound::Value(const Belief* belief) const {
	const vector<State*>& particles =
		static_cast<const ParticleBelief*>(belief)->particles();

	double value = 0;
	for (int i = 0; i < particles.size(); i++) {
		State* particle = particles[i];
		value += particle->weight * policy_[indexer_.GetIndex(particle)].value;
	}
	return value;
}

string MDPUpperBound::text() const {
	return "MDPUpperBound";
}

ostream& operator<<(ostream& os, const MDPUpperBound& b) {
	os << (&b)->text();
	return os;
}





} // namespace despot
