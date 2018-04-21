#ifndef DESPOT_H
#define DESPOT_H

#include <despot/core/solver.h>
#include <despot/core/pomdp.h>
#include <despot/core/belief.h>
#include <despot/core/node.h>
#include <despot/core/globals.h>
#include <despot/core/history.h>
#include <despot/random_streams.h>

namespace despot {

class DESPOT: public Solver {
friend class VNode;

protected:
	VNode* root_;
	SearchStatistics statistics_;

	ScenarioLowerBound* lower_bound_;
	ScenarioUpperBound* upper_bound_;

public:
	DESPOT(
      const DSPOMDP* model,
      ScenarioLowerBound* lb, ScenarioUpperBound* ub,
      Belief* belief = NULL);
	virtual ~DESPOT();

  /*
   * Find the best action for now
   */
	ValuedAction Search();

	void belief(Belief* b);
  /*
   * Update the history, the belief, and the belief inside the lower_bound_.
   */
	void Update(int action, OBS_TYPE obs);

	ScenarioLowerBound* lower_bound() const;
	ScenarioUpperBound* upper_bound() const;

  /*
   * Construct a tree with sampled particles or scenarios.
   * Each belief including all the associated particles is a VNode.
   * For each possible action, the tree will branch out from a VNode
   * to a QNode. Then at each QNode, we use streams to sample an
   * observation so that the QNode then transits to another VNode.
   */
	static VNode* ConstructTree(
      std::vector<State*>& particles, RandomStreams& streams,
      ScenarioLowerBound* lower_bound, ScenarioUpperBound* upper_bound,
      const DSPOMDP* model, History& history, double timeout,
      SearchStatistics* statistics = NULL);

protected:
  /*
   * 1st iteration:
   * From the VNode root, we expand it by executing all the possible actions.
   * After executing each action, we will get 1 QNode q.
   * From each q, we will sample an observation based on streams
   * and them expand q to get a list of VNodes v.
   * After this process, we will choose the QNode qstar with the highest upper
   * bound value and from qstar, we will further choose the VNode vstar
   * that has the highest Weighted Excess Utility.
   *
   * 2nd iteration:
   * Then we will expand from vstar ...
   *
   * We will stop when reaching the Globals::config.search_depth
   * or the WEU of the current VNode vstar <= 0.
   * We will return that VNode vstar.
   *
   * TODO: What is ExploitBlockers(cur); doing?
   */
	static VNode* Trial(VNode* root, RandomStreams& streams,
      ScenarioLowerBound* lower_bound, ScenarioUpperBound* upper_bound,
      const DSPOMDP* model, History& history,
      SearchStatistics* statistics = NULL);

  /*
   * Compute the bound weighted by the vnode->particles(),
   * based on the random number in streams, the history,
   * and vnode->particles().
   *
   * XXX: Note that lower_bound is the object to compute the bound,
   * not the specific value.
   * The specific values of bounds are stored in each VNode.
   */
	static void InitLowerBound(VNode* vnode, ScenarioLowerBound* lower_bound,
      RandomStreams& streams, History& history);
	static void InitUpperBound(VNode* vnode, ScenarioUpperBound* upper_bound,
      RandomStreams& streams, History& history);
  /*
   * Compute the lower and upper bounds, close the gap to 0 if upper < lower.
   */
	static void InitBounds(VNode* vnode, ScenarioLowerBound* lower_bound,
		ScenarioUpperBound* upper_bound, RandomStreams& streams, History& history);

  /*
   * Expand the vnode v into a list of QNode q's.
   * For each possible action a, we will expand v --a--> q.
   */
	static void Expand(
      VNode* vnode,
      ScenarioLowerBound* lower_bound, ScenarioUpperBound* upper_bound,
      const DSPOMDP* model, RandomStreams& streams, History& history);


  /*
   * Update VNode v1's lower and upper bounds based on the highest lower
   * bound and highest upper bound of its children QNodes.
   * v1 has multiple children because we can execute multiple actions at v1.
   */
	static void Update(VNode* vnode);
  /*
   * Update QNode q1's lower and upper bounds via the weighted average
   * of all of its children VNodes.
   * q1 has multiple children because q1's parent v1 has multiple particles
   * which will cause different observations at q1.
   */
	static void Update(QNode* qnode);
  /*
   * For each leaf VNode v1 of the tree, update its lower and upper bounds
   * based on the highest lower bound and highest upper bound of its children
   * QNodes.
   * v1 has multiple children because we can execute multiple actions at v1.
   *
   * Then for the parent of v1 - QNode q1, update its lower and upper bounds
   * via the weighted average of all of its children VNodes.
   * q1 has multiple children because q1's parent v1 has multiple particles
   * which will cause different observations at q1.
   *
   * Then update the parent of q1 ... (from leaf to root).
   */
	static void Backup(VNode* vnode);

	/*
   * Compute the difference between vnode->upper_bound()
   * and vnode->lower_bound().
   */
	static double Gap(VNode* vnode);

	double CheckDESPOT(const VNode* vnode, double regularized_value);
	double CheckDESPOTSTAR(const VNode* vnode, double regularized_value);
	void Compare();

	static void ExploitBlockers(VNode* vnode);
	static VNode* FindBlocker(VNode* vnode);

  /*
   * Expand the qnode q into a list of VNode v1's.
   * Note that the qnode q is obtained by executing an action a = qnode->edge()
   * from its parent as a VNode v0.
   * In order to get q's children v1's, we will execute a onto each particles
   * among v0's particle list.
   * Then we will use streams to sample an observation after executing a.
   * So in the end, we can compute step_reward - the average reward
   * we get by executing a across all the particles of v0.
   *
   * Then we will create children for q.
   * For each type of observation we get by executing a on v0's particles,
   * we will get a list of successive particles.
   * For each list of successive particles, we will create a new VNode v3
   * as a child of q. Then we will also InitBounds for v3.
   *
   * In the end, we will update the lower bound and upper bound of q
   * as bound_q = average_reward_of_executing_a + \sum(discounted_bound_v3)
   */
	static void Expand(
      QNode* qnode, ScenarioLowerBound* lower_bound,
      ScenarioUpperBound* upper_bound, const DSPOMDP* model,
      RandomStreams& streams, History& history);

	static VNode* Prune(VNode* vnode, int& pruned_action, double& pruned_value);
	static QNode* Prune(QNode* qnode, double& pruned_value);

  /*
   * WEU - Weighted Excess Utility - equation (11) in the jair17 paper.
   * It measures the difference between the current gap and the "expected"
   * gap if the target gap at the root is satisfied.
   *
   * Note that WEU is positively proporsional to vnode->Weight() which is the
   * total weight of all the particles in the vnode. If the Gap of vnode
   * can cover "the number of these particles" amounts of xi*Gap(root),
   * we are good and break the do-while loop of DESPOT::Trial.
   */
	static double WEU(VNode* vnode);
	static double WEU(VNode* vnode, double epsilon);
  /*
   * Return the vnode which has the highest WEU among all the children VNodes
   * of qnode.
   */
	static VNode* SelectBestWEUNode(QNode* qnode);
  /*
   * Return the qnode with the highest upper_bound among 
   * all the children qnodes of vnode.
   * The number of children that vnode has should be equivalent
   * to the number of possible actions.
   */
	static QNode* SelectBestUpperBoundNode(VNode* vnode);

  /*
   * Find the action (vnode->qnode) that can achieve the highest lower_bound
   * of qnode.
   */
	static ValuedAction OptimalAction(VNode* vnode);

	static ValuedAction Evaluate(VNode* root, std::vector<State*>& particles,
		RandomStreams& streams, POMCPPrior* prior, const DSPOMDP* model);
};

} // namespace despot

#endif
