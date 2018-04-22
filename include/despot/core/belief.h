#ifndef BELIEF_H
#define BELIEF_H

#include <vector>

#include <despot/util/random.h>
#include <despot/util/logging.h>
#include <despot/core/history.h>

namespace despot {

class State;
class StateIndexer;
class DSPOMDP;

/* =============================================================================
 * Belief class
 * =============================================================================*/

class Belief {
public:
	const DSPOMDP* model_;
	History history_;

public:
	Belief(const DSPOMDP* model);
	virtual ~Belief();

	virtual std::vector<State*> Sample(int num) const = 0;

  /*
   * Update the history by adding the action a and the observation o pair.
   * Update the particles by stepping each particle with the action
   * with randomly sampled observations o'.
   * Then use the prob of o to compute the prob of each new particle
   * and delete the particles whose prob=0 and which are terminated.
   * Now we have a newly weighted and filtered list of particles after
   * executing a and receiving o.
   *
   * If the list is empty, we resample.
   * Normalize the weights of the particles in the new list.
   */
	virtual void Update(int action, OBS_TYPE obs) = 0;

	virtual std::string text() const;
	friend std::ostream& operator<<(std::ostream& os, const Belief& x);
	virtual Belief* MakeCopy() const = 0;

  /*
   * Sample `num` number of particles from `belief`.
   *
   * When `num` > `belief.size()`:
   * e.g. `belief = [particle 0 with weight 0.2, particle 1 with weight 0.5,
   * particle 2 with weight 0.3] and num = 5.
   * Then this function will return a list of new particles
   * = [particle 0 with weight 0.2, particle 1 with weight 0.2,
   * particle 1 with weight 0.2, particle 1 with weight 0.2,
   * particle 2 with weight 0.2].
   *
   * When `num` <= `belief.size()`:
   * e.g. `belief = [particle 0 with weight 0.2, particle 1 with weight 0.5,
   * particle 2 with weight 0.3] and num = 2.
   * Then this function will return a list of new particles
   * = [particle 0 with weight 0.5, particle 1 with weight 0.5]
   *
   * What this function does is to sample a more evenly distributed set
   * to represent the probability mass inside the original particle list
   * `belief` as best as possible.
   */
	static std::vector<State*> Sample(
      int num, std::vector<State*> belief, const DSPOMDP* model);

  /*
   * We do Resample() because all of our original particles end up with
   * very low weights after the actions and observations in the history.
   * TODO: One reason could be that the scenarios we randomly selected
   * are not very appropriate.
   *
   * In Resample(), we sample a particle, step through the actions in the history,
   * and use random number to sample observations.
   * If prob is good, save it.
   * Remove all the particles with small weights.
   * Keep generating new particles until we have enough amount.
   */
  /*
   * I. Resampling by randomly going through the original particles
   * and save the particles that could still have a good probability
   * (i.e. survive) after the actions and observations in the past history.
   * Note that here we are using new random number in Step() so that
   * we might find different observations, comparing to the previous one.
   */
	static std::vector<State*> Resample(
      int num, const std::vector<State*>& belief,
      const DSPOMDP* model, History history, int hstart = 0);
  /*
   * II. Same with I.
   * Instead of using the `std::vector<State*>& belief` and `DSPOMDP* model`
   * from the arguments, here we directly take the vector of State* and
   * the Step() function from `Belief& belief`.
   */
	static std::vector<State*> Resample(
      int num, const Belief& belief, History history, int hstart = 0);

  /*
   * III. Resampling by using all the states from StateIndexer, which could
   * be all the possible states from the history for example.
   * Then for each state, we apply the pair of (action, obs)
   * from the last iteration. If the state after Step() has a good probability,
   * we will save it as a resampled particle.
   */
	static std::vector<State*> Resample(
      int num, const DSPOMDP* model,
      const StateIndexer* indexer, int action, OBS_TYPE obs);
};

/* =============================================================================
 * ParticleBelief class
 * =============================================================================*/

class ParticleBelief: public Belief {
protected:
	std::vector<State*> particles_;
	int num_particles_;
	Belief* prior_;
	bool split_;
	std::vector<State*> initial_particles_;
	const StateIndexer* state_indexer_;

public:
	ParticleBelief(std::vector<State*> particles, const DSPOMDP* model,
		Belief* prior = NULL, bool split = true);

	virtual ~ParticleBelief();
	void state_indexer(const StateIndexer* indexer);

	virtual const std::vector<State*>& particles() const;
	virtual std::vector<State*> Sample(int num) const;

	virtual void Update(int action, OBS_TYPE obs);

	virtual Belief* MakeCopy() const;

	virtual std::string text() const;
};

} // namespace despot

#endif
