#include <despot/solver/despot.h>
#include <despot/solver/pomcp.h>

using namespace std;

namespace despot {

DESPOT::DESPOT(const DSPOMDP* model, ScenarioLowerBound* lb, ScenarioUpperBound* ub, Belief* belief) :
	Solver(model, belief),
	root_(NULL), 
	lower_bound_(lb),
	upper_bound_(ub) {
	assert(model != NULL);
}

DESPOT::~DESPOT() {
}

ScenarioLowerBound* DESPOT::lower_bound() const {
	return lower_bound_;
}

ScenarioUpperBound* DESPOT::upper_bound() const {
	return upper_bound_;
}


VNode* DESPOT::Trial(
    VNode* root, RandomStreams& streams,
    ScenarioLowerBound* lower_bound, ScenarioUpperBound* upper_bound,
    const DSPOMDP* model, History& history, SearchStatistics* statistics)
{
  cout << "[DESPOT::Trial()]" << endl;

	VNode* cur = root;

	int hist_size = history.Size();
  cout << "hist_size=" << hist_size << endl;
  // TIGER: 0

	do
  {
		if (statistics != NULL
        && cur->depth() > statistics->longest_trial_length)
    {
			statistics->longest_trial_length = cur->depth();
		}

    cout << "At depth=" << cur->depth() << endl;
    // TIGER: 0

		ExploitBlockers(cur);

    cout << "Gap(cur)=" << Gap(cur) << endl;
    // TIGER: 220
		if (Gap(cur) == 0)
    {
			break;
		}

		if (cur->IsLeaf())
    {
			double start = clock();
			Expand(cur, lower_bound, upper_bound, model, streams, history);

			if (statistics != NULL)
      {
				statistics->time_node_expansion
            += (double) (clock() - start) / CLOCKS_PER_SEC;
				statistics->num_expanded_nodes++;
				statistics->num_tree_particles += cur->particles().size();
			}
		}

    root->PrintTree();
    // TIGER:
    // a - action
    // o - observation
    // d - default value
    // l - lower bound
    // u - upper bound
    // r - totol weighted step_reward for 1 step
    // w - total particle weight
    // (d:-20 l:-20, u:200, w:1, weu:11)
    // a=0: (d:-60.92, l:-60.92, u:148.08, r:-41.92)
    // | o=2: (d:-19 l:-19, u:190, w:1, weu:-1.42109e-13)
    // a=1: (d:-67.08, l:-67.08, u:141.92, r:-48.08)
    // | o=2: (d:-19 l:-19, u:190, w:1, weu:-1.42109e-13)
    // a=2: (d:-20, l:-20, u:189, r:-1)
    // | o=0: (d:-9.462 l:-9.462, u:94.62, w:0.498, weu:-7.10543e-14)
    // | o=1: (d:-9.538 l:-9.538, u:95.38, w:0.502, weu:-8.52651e-14)
    //
    // XXX:
    // The upper bound of the vnode at a=2 = u1 = 189.
    // The step_reward of the vnode at a=2 = r = -1.
    // The upper bound of the qnode when a=2 and o=0 = u2 = 94.62.
    // The upper bound of the qnode when a=2 and o=1 = u3 = 95.38.
    // Then u1 = u2 + u3 + r.
    // Note that we don't need to do u1 = 50%*u2 + 50%*u3 + r is because
    // we have already weighted u2 and u3 inside InitBounds.

    root->PrintPolicyTree();
    // 0x130d4d0-a=2
    // | o=0: 0x130d9a0-a=2
    // | o=1: 0x130e720-a=2

		double start = clock();
		QNode* qstar = SelectBestUpperBoundNode(cur);
    cout << "The QNode with the best upper_bound_ = " << *qstar << endl;
    // TIGER:
    // #QNode: [lower_bound_=-20, upper_bound_=189, step_reward=-1,
    // utility_upper_bound=189]#
    // Note that the upper_bound of the qnode is
    // reward + value of successive vnode from this qnode.

		VNode* next = SelectBestWEUNode(qstar);
    cout << "The vnode with best WEU = " << *next << endl;
    // TIGER:
    // @VNode: [depth_=1, lower_bound_=-9.462, upper_bound_=94.62]@

		if (statistics != NULL) {
			statistics->time_path += (clock() - start) / CLOCKS_PER_SEC;
		}

		if (next == NULL) {
			break;
		}

		cur = next;
		history.Add(qstar->edge(), cur->edge());

    cout << "cur=" << *cur << "\nWEN(cur)=" << WEU(cur)
        << ", depth=" << cur->depth() << endl;
	} while (cur->depth() < Globals::config.search_depth && WEU(cur) > 0);

  // resize history to the original size - hist_size
	history.Truncate(hist_size);

	return cur;
}

void DESPOT::ExploitBlockers(VNode* vnode) {
  cout << "[DESPOT::ExploitBlockers()]" << endl;
  cout << "Globals::config.pruning_constant="
    << Globals::config.pruning_constant << endl;

	if (Globals::config.pruning_constant <= 0) {
		return;
	}

	VNode* cur = vnode;
	while (cur != NULL) {
		VNode* blocker = FindBlocker(cur);

		if (blocker != NULL) {
			if (cur->parent() == NULL || blocker == cur) {
				double value = cur->default_move().value;
				cur->lower_bound(value);
				cur->upper_bound(value);
				cur->utility_upper_bound = value;
			} else {
				const map<OBS_TYPE, VNode*>& siblings =
					cur->parent()->children();
				for (map<OBS_TYPE, VNode*>::const_iterator it = siblings.begin();
					it != siblings.end(); it++) {
					VNode* node = it->second;
					double value = node->default_move().value;
					node->lower_bound(value);
					node->upper_bound(value);
					node->utility_upper_bound = value;
				}
			}

			Backup(cur);

			if (cur->parent() == NULL) {
				cur = NULL;
			} else {
				cur = cur->parent()->parent();
			}
		} else {
			break;
		}
	}
}

VNode* DESPOT::ConstructTree(
    vector<State*>& particles, RandomStreams& streams,
    ScenarioLowerBound* lower_bound, ScenarioUpperBound* upper_bound,
    const DSPOMDP* model, History& history, double timeout,
    SearchStatistics* statistics)
{
  cout << "[DESPOT::ConstructTree()]" << endl;

	if (statistics != NULL) {
		statistics->num_particles_before_search = model->NumActiveParticles();
	}

  cout << model->NumActiveParticles() << endl;
  for (int i = 0; i < particles.size(); i++) {
    particles[i]->scenario_id = i;
  }

  // TIGER:
  // model->NumActiveParticles() = 4598
  // for (int i = 0; i < particles.size(); i++) {
    // cout << "[" << i << "]" << *particles[i] << endl;
  // }
  // TIGER:
  // particles.size() = 500

	VNode* root = new VNode(particles);
  // TIGER:
  // root=@VNode: [depth_=0, lower_bound_=8.51073e-317,
  // upper_bound_=1.58101e-322]@
  cout << "root=" << *root << endl;

  logd << "[DESPOT::ConstructTree] START - Initializing lower and "
    "upper bounds at the root node.";
	InitBounds(root, lower_bound, upper_bound, streams, history);

    exit(0);
  logd << "[DESPOT::ConstructTree] END - Initializing lower and "
    "upper bounds at the root node.";

	if (statistics != NULL) {
		statistics->initial_lb = root->lower_bound();
		statistics->initial_ub = root->upper_bound();
	}
  cout << "After initializing bonuds, statistics=\n" << *statistics << endl;
  // After initializing bonuds, statistics=
  // Initial bounds: (-20, 200)
  // Final bounds: (-inf, inf)
  // Time (CPU s): path / expansion / backup / total = 0 / 0 / 0 / 0
  // Trials: no. / max length = 0 / 0
  // # nodes: expanded / total / policy = 0 / 0 / 0
  // # particles: initial / final / tree = 4598 / 0 / 0

	double used_time = 0;
	int num_trials = 0;
	do
  {
		double start = clock();
		VNode* cur = Trial(root, streams, lower_bound, upper_bound,
        model, history, statistics);
		used_time += double(clock() - start) / CLOCKS_PER_SEC;

		start = clock();
		Backup(cur);
		if (statistics != NULL)
    {
			statistics->time_backup += double(clock() - start) / CLOCKS_PER_SEC;
		}
		used_time += double(clock() - start) / CLOCKS_PER_SEC;

		num_trials++;
	}
  while (used_time * (num_trials + 1.0) / num_trials < timeout
      && (root->upper_bound() - root->lower_bound()) > 1e-6);

	if (statistics != NULL)
  {
		statistics->num_particles_after_search = model->NumActiveParticles();
		statistics->num_policy_nodes = root->PolicyTreeSize();
		statistics->num_tree_nodes = root->Size();
		statistics->final_lb = root->lower_bound();
		statistics->final_ub = root->upper_bound();
		statistics->time_search = used_time;
		statistics->num_trials = num_trials;
	}

	return root;
}

void DESPOT::Compare() {
	vector<State*> particles = belief_->Sample(Globals::config.num_scenarios);
	SearchStatistics statistics;

	RandomStreams streams = RandomStreams(Globals::config.num_scenarios,
		Globals::config.search_depth);

	VNode* root = ConstructTree(particles, streams, lower_bound_, upper_bound_,
		model_, history_, Globals::config.time_per_move, &statistics);

	CheckDESPOT(root, root->lower_bound());
	CheckDESPOTSTAR(root, root->lower_bound());
	delete root;
}

void DESPOT::InitLowerBound(
    VNode* vnode, ScenarioLowerBound* lower_bound,
    RandomStreams& streams, History& history)
{
  cout << "[DESPOT::InitLowerBound()]" << endl;

  cout << "Old vnode=" << *vnode << endl;
  // TIGER:
  // lower_bound is TrivialParticleLowerBound.
  // Old vnode=@VNode:
  // [depth_=0, lower_bound_=1.05142e-316, upper_bound_=1.58101e-322]@

	streams.position(vnode->depth());
	
  // For TrivialParticleLowerBound,
  // return the max min_reward action across all actions with its value
	ValuedAction move = lower_bound->Value(vnode->particles(), streams, history);
  cout << "At depth=" << vnode->depth() << ", lower_bound (action, value) ="
    << move << endl;

  // TIGER:
  // The max min_reward action is LISTEN. Its min_reward = -1.
  // So here -1 / 1-(discount=0.95) = -20
  // At depth=0, lower_bound (action, value) =(2, -20)

  // We assume the vnode->particles() has depth 0 in lower_bound->Value(),
  // so now we need to multiple the discount again.
	move.value *= Globals::Discount(vnode->depth());

  cout << "After discounted, (action, value)=" << move << endl;
  // TIGER:
  // After discounted, (action, value)=(2, -20)

	vnode->default_move(move);
	vnode->lower_bound(move.value);

  cout << "New vnode=" << *vnode << endl;
  // TIGER:
  // New vnode=@VNode:
  // [depth_=0, lower_bound_=-20, upper_bound_=1.58101e-322]@
}


void DESPOT::InitUpperBound(
    VNode* vnode, ScenarioUpperBound* upper_bound,
    RandomStreams& streams, History& history)
{
  cout << "[DESPOT::InitUpperBound()]" << endl;

  cout << "Old vnode=" << *vnode << endl;
  // TIGER:
  // lower_bound is TrivialParticleLowerBound.
  // Old vnode=@VNode:
  // [depth_=0, lower_bound_=-20, upper_bound_=1.58101e-322]@

	streams.position(vnode->depth());

  // For TrivialParticleLowerBound,
  // return the max_reward action across all actions with its value
	double upper = upper_bound->Value(vnode->particles(), streams, history);

  cout<< "here111111111" << endl;
  exit(0);
  cout << "At depth=" << vnode->depth() << ", upper_bound value ="
    << upper << endl;
  // TIGER:
  // The max_reward action is LEFT and RIGHT. Its max_reward = 10;
  // So here 10 / 1-(discount=0.95) = 200
  // At depth=0, upper_bound value =200

  // We assume the vnode->particles() has depth 0 in upper_bound->Value(),
  // so now we need to multiple the discount again.
	vnode->utility_upper_bound = upper * Globals::Discount(vnode->depth());

  cout << "After discounted, value=" << vnode->utility_upper_bound << endl;
  // TIGER:
  // After discounted, value=200

	upper = upper * Globals::Discount(vnode->depth())\
          - Globals::config.pruning_constant;
	vnode->upper_bound(upper);

  cout << "New vnode=" << *vnode << endl;
  // TIGER:
  // New vnode=@VNode: [depth_=0, lower_bound_=-20, upper_bound_=200]@
}

void DESPOT::InitBounds(VNode* vnode, ScenarioLowerBound* lower_bound,
	ScenarioUpperBound* upper_bound, RandomStreams& streams, History& history) {
  cout << "[DESPOT::InitBounds()]" << endl;

  InitLowerBound(vnode, lower_bound, streams, history);

  cout<< "here111111111" << endl;
  exit(0);
	InitUpperBound(vnode, upper_bound, streams, history);

	if (vnode->upper_bound() < vnode->lower_bound()
		// close gap because no more search can be done on leaf node
		|| vnode->depth() == Globals::config.search_depth - 1) {
		vnode->upper_bound(vnode->lower_bound());
	}
}

ValuedAction DESPOT::Search()
{
  cout<<"[DESPOT::Search()]"<<endl;

  // TIGER:
  // Sorted pdf for 4096 particles:
  //   LEFT = 0.5
  //   RIGHT = 0.5
  // cout << *belief_ << endl;

	if (logging::level() >= logging::DEBUG)
    model_->PrintBelief(*belief_);

  // Return a random action if no time is allocated for planning
	if (Globals::config.time_per_move <= 0)
  {
		return ValuedAction(
        Random::RANDOM.NextInt(model_->NumActions()),
        Globals::NEG_INFTY);
  }

	double start = get_time_second();
	vector<State*> particles = belief_->Sample(Globals::config.num_scenarios);
  cout << "we sample " << Globals::config.num_scenarios << " scenarios.";
  // for (int i = 0; i < particles.size(); i ++)
  // {
    // cout << "particles[" << i << "]: " << *particles[i] << endl;
  // }
  // TIGER:
  // [0]: (state_id = -1, weight = 0.002, text = LEFT)
  // [1]: (state_id = -1, weight = 0.002, text = RIGHT)
  // [2]: (state_id = -1, weight = 0.002, text = RIGHT)
  // [3]: (state_id = -1, weight = 0.002, text = RIGHT)
  // [4]: (state_id = -1, weight = 0.002, text = LEFT)
  // [5]: (state_id = -1, weight = 0.002, text = RIGHT)
  // [6]: (state_id = -1, weight = 0.002, text = RIGHT)
  // [7]: (state_id = -1, weight = 0.002, text = RIGHT)
  // [8]: (state_id = -1, weight = 0.002, text = LEFT)
  logi << "[DESPOT::Search] Time for sampling " << particles.size()
		<< " particles: " << (get_time_second() - start) << "s" << endl;

  // XXX: In the beginning inside [SimpleRockSample::InitialBelief()],
  // we Allocate() 2 particles,
  // and then use `new ParticleBelief(particles, this);` to split
  // the 2 into 4096 particles. But we don't remove them from the memory,
  // so now we have 4098 particles.
  // Now we sample 500 particles from the 4096 particles in ParticleBelief,
  // again we will not remove the previous 4098 particles from the memory,
  // so now we have 4098 + 500 = 4598 particles.
  //
  // for debug
  // cout << model_->NumActiveParticles() << endl;
  // exit(0);

	statistics_ = SearchStatistics();
  // cout << statistics_ << endl;
  // TIGER:
  // Initial bounds: (-inf, inf)
  // Final bounds: (-inf, inf)
  // Time (CPU s): path / expansion / backup / total = 0 / 0 / 0 / 0
  // Trials: no. / max length = 0 / 0
  // # nodes: expanded / total / policy = 0 / 0 / 0
  // # particles: initial / final / tree = 0 / 0 / 0

	start = get_time_second();
	static RandomStreams streams = RandomStreams(Globals::config.num_scenarios,
		Globals::config.search_depth);
  cout << "create num_scenarios=" << Globals::config.num_scenarios
    << " RandomStreams and each has length search_depth="
    << Globals::config.search_depth<< endl;
  // cout << streams << endl;
  // TIGER:
  // Create num_scenarios=500 RandomStreams
  // and each has length search_depth=90.
  // So streams = [500 x 90]:
  // Stream 0: 0.900913 0.629146 0.736037 0.730071 0.431576 ... 0.401434
  // Stream 1: 0.357293 0.184858 0.107582 0.610372 0.336558 ... 0.871826 
  // ...
  // Stream 499: 0.357293 0.184858 0.107582 0.610372 0.336558 ... 0.871826 

  // TIGER:
  // upper_bound_ = TrivialParticleUpperBound
  // We will fail to cast it to LookaheadUpperBound, so ub = NULL
	LookaheadUpperBound* ub = dynamic_cast<LookaheadUpperBound*>(upper_bound_);

  // Avoid using new streams for LookaheadUpperBound
	if (ub != NULL)
  { 
		static bool initialized = false;
		if (!initialized ) {
			lower_bound_->Init(streams);
			upper_bound_->Init(streams);
			initialized = true;
		}
    cout << "ub != NULL" << endl;
	} else {
		streams = RandomStreams(Globals::config.num_scenarios,
			Globals::config.search_depth);
    // This is ScenarioUpperBound::Init(), which does nothing.
		lower_bound_->Init(streams);
		upper_bound_->Init(streams);
    cout << "ub == NULL" << endl;
	}

  // TIGER:
  // 500 particles, [500 x 90] streams, default TrivialParticleLowerBound,
  // default TrivialParticleUpperBound, empty history_, default statistics_.
	root_ = ConstructTree(particles, streams, lower_bound_, upper_bound_,
      model_, history_, Globals::config.time_per_move, &statistics_);
	logi << "[DESPOT::Search] Time for tree construction: "
		<< (get_time_second() - start) << "s" << endl;

	start = get_time_second();
	root_->Free(*model_);
	logi << "[DESPOT::Search] Time for freeing particles in search tree: "
		<< (get_time_second() - start) << "s" << endl;

	ValuedAction astar = OptimalAction(root_);
	start = get_time_second();
	delete root_;

	logi << "[DESPOT::Search] Time for deleting tree: "
		<< (get_time_second() - start) << "s" << endl;
	logi << "[DESPOT::Search] Search statistics:" << endl << statistics_
		<< endl;

	return astar;
}

double DESPOT::CheckDESPOT(const VNode* vnode, double regularized_value) {
	cout
		<< "--------------------------------------------------------------------------------"
		<< endl;

	const vector<State*>& particles = vnode->particles();
	vector<State*> copy;
	for (int i = 0; i < particles.size(); i ++) {
		copy.push_back(model_->Copy(particles[i]));
	}
	VNode* root = new VNode(copy);

	double pruning_constant = Globals::config.pruning_constant;
	Globals::config.pruning_constant = 0;

	RandomStreams streams = RandomStreams(Globals::config.num_scenarios,
		Globals::config.search_depth);

	streams.position(0);
	InitBounds(root, lower_bound_, upper_bound_, streams, history_);

	double used_time = 0;
	int num_trials = 0, prev_num = 0;
	double pruned_value;
	do {
		double start = clock();
		VNode* cur = Trial(root, streams, lower_bound_, upper_bound_, model_, history_);
		num_trials++;
		used_time += double(clock() - start) / CLOCKS_PER_SEC;

		start = clock();
		Backup(cur);
		used_time += double(clock() - start) / CLOCKS_PER_SEC;

		if (double(num_trials - prev_num) > 0.05 * prev_num) {
			int pruned_action;
			Globals::config.pruning_constant = pruning_constant;
			VNode* pruned = Prune(root, pruned_action, pruned_value);
			Globals::config.pruning_constant = 0;
			prev_num = num_trials;

			pruned->Free(*model_);
			delete pruned;

			cout << "# trials = " << num_trials << "; target = "
				<< regularized_value << ", current = " << pruned_value
				<< ", l = " << root->lower_bound() << ", u = "
				<< root->upper_bound() << "; time = " << used_time << endl;

			if (pruned_value >= regularized_value) {
				break;
			}
		}
	} while (true);

	cout << "DESPOT: # trials = " << num_trials << "; target = "
		<< regularized_value << ", current = " << pruned_value << ", l = "
		<< root->lower_bound() << ", u = " << root->upper_bound() << "; time = "
		<< used_time << endl;
	Globals::config.pruning_constant = pruning_constant;
	cout
		<< "--------------------------------------------------------------------------------"
		<< endl;

	root->Free(*model_);
	delete root;

	return used_time;
}

double DESPOT::CheckDESPOTSTAR(const VNode* vnode, double regularized_value) {
	cout
		<< "--------------------------------------------------------------------------------"
		<< endl;

	const vector<State*>& particles = vnode->particles();
	vector<State*> copy;
	for (int i = 0; i < particles.size(); i++) {
		copy.push_back(model_->Copy(particles[i]));
	}
	VNode* root = new VNode(copy);

	RandomStreams streams = RandomStreams(Globals::config.num_scenarios,
		Globals::config.search_depth);
	InitBounds(root, lower_bound_, upper_bound_, streams, history_);

	double used_time = 0;
	int num_trials = 0;
	do {
		double start = clock();
		VNode* cur = Trial(root, streams, lower_bound_, upper_bound_, model_, history_);
		num_trials++;
		used_time += double(clock() - start) / CLOCKS_PER_SEC;

		start = clock();
		Backup(cur);
		used_time += double(clock() - start) / CLOCKS_PER_SEC;
	} while (root->lower_bound() < regularized_value);

	cout << "DESPOT: # trials = " << num_trials << "; target = "
		<< regularized_value << ", current = " << root->lower_bound()
		<< ", l = " << root->lower_bound() << ", u = " << root->upper_bound()
		<< "; time = " << used_time << endl;
	cout
		<< "--------------------------------------------------------------------------------"
		<< endl;

	root->Free(*model_);
	delete root;

	return used_time;
}

VNode* DESPOT::Prune(VNode* vnode, int& pruned_action, double& pruned_value) {
	vector<State*> empty;
	VNode* pruned_v = new VNode(empty, vnode->depth(), NULL,
		vnode->edge());

	vector<QNode*>& children = vnode->children();
	int astar = -1;
	double nustar = Globals::NEG_INFTY;
	QNode* qstar = NULL;
	for (int i = 0; i < children.size(); i++) {
		QNode* qnode = children[i];
		double nu;
		QNode* pruned_q = Prune(qnode, nu);

		if (nu > nustar) {
			nustar = nu;
			astar = qnode->edge();

			if (qstar != NULL) {
				delete qstar;
			}

			qstar = pruned_q;
		} else {
			delete pruned_q;
		}
	}

	if (nustar < vnode->default_move().value) {
		nustar = vnode->default_move().value;
		astar = vnode->default_move().action;
		delete qstar;
	} else {
		pruned_v->children().push_back(qstar);
		qstar->parent(pruned_v);
	}

	pruned_v->lower_bound(vnode->lower_bound()); // for debugging
	pruned_v->upper_bound(vnode->upper_bound());

	pruned_action = astar;
	pruned_value = nustar;

	return pruned_v;
}

QNode* DESPOT::Prune(QNode* qnode, double& pruned_value) {
	QNode* pruned_q = new QNode((VNode*) NULL, qnode->edge());
	pruned_value = qnode->step_reward - Globals::config.pruning_constant;
	map<OBS_TYPE, VNode*>& children = qnode->children();
	for (map<OBS_TYPE, VNode*>::iterator it = children.begin();
		it != children.end(); it++) {
		int astar;
		double nu;
		VNode* pruned_v = Prune(it->second, astar, nu);
		if (nu == it->second->default_move().value) {
			delete pruned_v;
		} else {
			pruned_q->children()[it->first] = pruned_v;
			pruned_v->parent(pruned_q);
		}
		pruned_value += nu;
	}

	pruned_q->lower_bound(qnode->lower_bound()); // for debugging
	pruned_q->upper_bound(qnode->upper_bound()); // for debugging

	return pruned_q;
}


ValuedAction DESPOT::OptimalAction(VNode* vnode)
{
  cout << "[DESPOT::OptimalAction]" << endl;

	ValuedAction astar(-1, Globals::NEG_INFTY);
	for (int action = 0; action < vnode->children().size(); action++)
  {
		QNode* qnode = vnode->Child(action);
		if (qnode->lower_bound() > astar.value)
    {
			astar = ValuedAction(action, qnode->lower_bound());
		}
	}

  // if default_move is better, we do default_move.
	if (vnode->default_move().value > astar.value)
  {
		astar = vnode->default_move();
	}

	return astar;
}

double DESPOT::Gap(VNode* vnode) {
	return (vnode->upper_bound() - vnode->lower_bound());
}

double DESPOT::WEU(VNode* vnode) {
	return WEU(vnode, Globals::config.xi);
}

// Can pass root as an argument, but will not affect performance much
double DESPOT::WEU(VNode* vnode, double xi) {
	VNode* root = vnode;
	while (root->parent() != NULL) {
		root = root->parent()->parent();
	}
	return Gap(vnode) - xi * vnode->Weight() * Gap(root);
}

VNode* DESPOT::SelectBestWEUNode(QNode* qnode)
{
  cout << "[DESPOT::SelectBestWEUNode]" << endl;
	double weustar = Globals::NEG_INFTY;
	VNode* vstar = NULL;
	map<OBS_TYPE, VNode*>& children = qnode->children();
	for (map<OBS_TYPE, VNode*>::iterator it = children.begin();
      it != children.end(); it++)
  {
		VNode* vnode = it->second;

		double weu = WEU(vnode);
		if (weu >= weustar) {
			weustar = weu;
      // node.vstar is itself if node == VNode
      // it is NULL if node == QNode
			vstar = vnode->vstar;
		}
	}
	return vstar;
}

QNode* DESPOT::SelectBestUpperBoundNode(VNode* vnode)
{
  cout << "[DESPOT::SelectBestUpperBoundNode]" << endl;

	int astar = -1;
	double upperstar = Globals::NEG_INFTY;
	for (int action = 0; action < vnode->children().size(); action++)
  {
		QNode* qnode = vnode->Child(action);
    // cout << qnode->upper_bound() << endl;

		if (qnode->upper_bound() > upperstar) {
			upperstar = qnode->upper_bound();
			astar = action;
		}
	}
	assert(astar >= 0);
	return vnode->Child(astar);
}

void DESPOT::Update(VNode* vnode)
{
  // cout << "[DESPOT::Update for VNode]" << endl;
  // cout << "Before update the bounds, vnode=" << *vnode << endl;

	if (vnode->IsLeaf())
  {
		return;
	}

  // default_move was assigned in DESPOT::InitLowerBound().
  // It is the action associated with the lower_bound_ of a VNode.
	double lower = vnode->default_move().value;
	double upper = vnode->default_move().value;
  // XXX: This is U() in the jair17 paper.
	double utility_upper = Globals::NEG_INFTY;

	for (int action = 0; action < vnode->children().size(); action++)
  {
		QNode* qnode = vnode->Child(action);

    // qnode's bound = reward(vnode->qnode) + discounted_sum of
    // the lower_bounds of all qnode's children.
		lower = max(lower, qnode->lower_bound());
    // TODO: Why upper is still max? Maybe we want the reward to be high,
    // so max
    // Maybe we have to make sure all the upper bounds of qnode is included
    // by the upper variable?
		upper = max(upper, qnode->upper_bound());
		utility_upper = max(utility_upper, qnode->utility_upper_bound);

    // cout << *qnode << endl;
	}

  // Shrink the gap between the lower and upper bounds for vnode
  // by choosing the small upper and larger lower bounds.
	if (lower > vnode->lower_bound())
  {
		vnode->lower_bound(lower);
	}
  // TODO: Why now use the smaller upper? not consistent with the previous TODO
	if (upper < vnode->upper_bound())
  {
		vnode->upper_bound(upper);
	}
	if (utility_upper < vnode->utility_upper_bound)
  {
		vnode->utility_upper_bound = utility_upper;
	}

  // cout << "After update the bounds, vnode=" << *vnode << endl;
  // TIGER:
  // Before update the bounds, vnode=@VNode:
  // [depth_=0, lower_bound_=-20, upper_bound_=200]@
  // #QNode: [lower_bound_=-61.14, upper_bound_=147.86,
  // step_reward=-42.14, utility_upper_bound=147.86]#
  // #QNode: [lower_bound_=-66.86, upper_bound_=142.14,
  // step_reward=-47.86, utility_upper_bound=142.14]#
  // #QNode: [lower_bound_=-20, upper_bound_=189,
  // step_reward=-1, utility_upper_bound=189]#
  // After update the bounds, vnode=@VNode:
  // [depth_=0, lower_bound_=-20, upper_bound_=189]@
  //
  // XXX: The intuition here is that the bounds of root VNode are estimated
  // roughly and naively at first by only using the default bounds.
  // When we expand root VNode v1 to get a list of QNodes q2,
  // we actually reduce our uncertainty in the bounds of v1 because
  // we know a part of the bounds of v1 without any uncertainty.
  // This part is the reward(root, QNodes).
  // Whenever we expand a node and get rewards, we shrink the gap between the
  // bounds.
  // Therefore, now if we use the children QNodes q2 to update the bounds
  // of VNode v1, we will be able to shorter the gap between the bounds of v1.
}

void DESPOT::Update(QNode* qnode)
{
  cout << "[DESPOT::Update for QNode]" << endl;
  // cout << "Before update the bounds, qnode=" << *qnode << endl;

	double lower = qnode->step_reward;
	double upper = qnode->step_reward;
	double utility_upper = qnode->step_reward
		+ Globals::config.pruning_constant;

	map<OBS_TYPE, VNode*>& children = qnode->children();
	for (map<OBS_TYPE, VNode*>::iterator it = children.begin();
      it != children.end(); it++)
  {
		VNode* vnode = it->second;

		lower += vnode->lower_bound();
		upper += vnode->upper_bound();
		utility_upper += vnode->utility_upper_bound;

    // cout << *vnode << endl;
	}

	if (lower > qnode->lower_bound()) {
		qnode->lower_bound(lower);
	}
	if (upper < qnode->upper_bound()) {
		qnode->upper_bound(upper);
	}
	if (utility_upper < qnode->utility_upper_bound) {
		qnode->utility_upper_bound = utility_upper;
	}
  // cout << "After update the bounds, qnode=" << *qnode << endl;

  // TIGER:
  // Before update the bounds, qnode=#QNode:
  // [lower_bound_=-20, upper_bound_=189, step_reward=-1,
  // utility_upper_bound=189]#
  // @VNode: [depth_=1, lower_bound_=-8.626, upper_bound_=86.26]@
  // @VNode: [depth_=1, lower_bound_=-10.374, upper_bound_=103.74]@
  // After update the bounds, qnode=#QNode:
  // [lower_bound_=-20, upper_bound_=189, step_reward=-1,
  // utility_upper_bound=189]#
  //
  // XXX: The intuition here is that if we ever update the bounds of
  // the children of qnode, then it is necessary to re-compute the bounds
  // of the qnode.
  // However, if we never update the bounds of the bounds of the children
  // of qnode, there is no need to update the bounds of qnode here.
}

void DESPOT::Backup(VNode* vnode)
{
  cout << "[DESPOT::Backup]" << endl;

	logd << "- Backup " << *vnode << " at depth " << vnode->depth() << endl;
  // TIGER:
  // Backup @VNode: [depth_=1, lower_bound_=-9.462,
  // upper_bound_=94.62]@ at depth 1

	int iter = 0;
	while (true)
  {
		logd << " Iter " << iter << " " << vnode << endl;

		Update(vnode);

		QNode* parentq = vnode->parent();
		if (parentq == NULL) {
			break;
		}

		Update(parentq);

		logd << " Updated Q-node to (" << parentq->lower_bound() << ", "
			<< parentq->upper_bound() << ")" << endl;

		vnode = parentq->parent();
		iter++;
	}
	logd << "* Backup complete!" << endl;
}

VNode* DESPOT::FindBlocker(VNode* vnode) {
	VNode* cur = vnode;
	int count = 1;
	while (cur != NULL) {
		if (cur->utility_upper_bound - count * Globals::config.pruning_constant
			<= cur->default_move().value) {
			break;
		}
		count++;
		if (cur->parent() == NULL) {
			cur = NULL;
		} else {
			cur = cur->parent()->parent();
		}
	}
	return cur;
}

void DESPOT::Expand(
    VNode* vnode,
    ScenarioLowerBound* lower_bound, ScenarioUpperBound* upper_bound,
    const DSPOMDP* model, RandomStreams& streams,
    History& history)
{
  cout << "[DESPOT::Expand() for VNode]" << endl;

  // Now children is empty.
  // Note that the children here is a reference to the
  // children_ inside the VNode,
  // So we will update the children inside VNode here.
	vector<QNode*>& children = vnode->children();

  logd << "- Expanding vnode " << *vnode << endl;
  // Tiger:
  // Expanding vnode @VNode: [depth_=0, lower_bound_=-20, upper_bound_=200]@

  // Here we will expand all the possible actions:
	for (int action = 0; action < model->NumActions(); action++)
  {
    logd << " actions[" << action << "]" << endl;

    // The qnode's parent is vnode, and the edge between the qnode and vnode
    // is the action.
		QNode* qnode = new QNode(vnode, action);
		children.push_back(qnode);

    // This is DESPOT::Expand(QNode* qnode, ... ) for a QNode.
		Expand(qnode, lower_bound, upper_bound, model, streams, history);
	}
	logd << "* Expansion complete!" << endl;
}

void DESPOT::Expand(
    QNode* qnode, ScenarioLowerBound* lb,
    ScenarioUpperBound* ub, const DSPOMDP* model,
    RandomStreams& streams,
    History& history)
{
  cout << "\n[DESPOT::Expand() for QNode]" << endl;
  cout << "Before creating new vnodes as children, qnode=" << *qnode << endl;
  // TIGER:
  // Before creating new vnodes as children, qnode=#QNode:
  // [lower_bound_=1.53832e+256, upper_bound_=4.71221e+257,
  // step_reward=6.96794e+98, utility_upper_bound=9.10979e+227]#

	VNode* parent = qnode->parent();

  // setting position inside streams
	streams.position(parent->depth());

  // Now children is empty.
  // Note that the children here is a reference to the
  // children_ inside the qnode.
  // So we will update children here to update qnode.
	map<OBS_TYPE, VNode*>& children = qnode->children();

  // TIGER: the sample of 500 particles
	const vector<State*>& particles = parent->particles();

	double step_reward = 0;

  // Here we will step all the particles forward for 1 step,
  // and sort the newly updated particles based on the observations
  // that they receive within the 1 step.
  // Then we will save the new sorted particles in partitions.
  // In the end, we will compute the reward averaged across all the particles
  // in step_reward.

	// Partition particles by observation
	map<OBS_TYPE, vector<State*> > partitions;
	OBS_TYPE obs;
	double reward;
	for (int i = 0; i < particles.size(); i++)
  {
		State* particle = particles[i];

		// logd << " Original: " << *particle << endl;
		// cout << " Original particles[" << i << "] = " << *particle << endl;
    // TIGER:
    // Original particles[0] = (state_id = -1, weight = 0.002, text = LEFT)

		State* copy = model->Copy(particle);

		// logd << " Before step: " << *copy << endl;
		// cout << "Before step, copy=" << *copy << endl
      // << "copy->scenario_id=" << copy->scenario_id
      // << ", parent->depth()=" << parent->depth()
      // << ", streams_[scenario_id][depth]="
      // << streams.Entry(copy->scenario_id)
      // << ", action=" << qnode->edge() << endl;
    // TIGER:
    // Before step, copy=(state_id = -1, weight = 0.002, text = LEFT)
    // copy->scenario_id=0, parent->depth()=0,
    // streams_[scenario_id][depth]=0.170837, action=0

		bool terminal = model->Step(
        *copy, streams.Entry(copy->scenario_id),
        qnode->edge(), reward, obs);

    // cout << "After step, copy=" << *copy << endl
      // << "reward=" << reward << ", observation=" << obs
      // << ", weighted reward=" << reward * copy->weight << endl;
    // TIGER:
    // After step, copy=(state_id = -1, weight = 0.002, text = RIGHT)
    // reward=-100, observation=2, weighted reward=-0.2

		step_reward += reward * copy->weight;

		// logd << " After step: " << *copy << " " << (reward * copy->weight)
			// << " " << reward << " " << copy->weight << endl;

		if (!terminal)
    {
			partitions[obs].push_back(copy);
		}
    else
    {
			model->Free(copy);
		}
	}

  cout << "step_reward averaged across all the particles = "
    << step_reward << endl;
  // TIGER:
  // step_reward averaged across all the particles = -43.24
  // XXX: action0 = LEFT.
  // If there could be infinite number of particles inside particles,
  // with 50% state=LEFT and 50% state=RIGHT,
  // then the expected step_reward = 0.5 * -100 + 0.5 * 10 = -45.
  // Since we only have 500 particles, -43.24 makes sense.

	step_reward = Globals::Discount(parent->depth()) * step_reward
    //pruning_constant is used for regularization
    - Globals::config.pruning_constant;

  cout << "At depth=" << parent->depth()
    << ", Globals::Discount(depth)=" << Globals::Discount(parent->depth())
    << ", Globals::config.pruning_constant="
    << Globals::config.pruning_constant
    << ", step_reward=" << step_reward << endl;
  // At depth=0, Globals::Discount(depth)=1,
  // Globals::config.pruning_constant=0, step_reward=-43.24

  // XXX: Here we add step_reward to lower_bound:
  // Q_value = Reward + gamma * Value
	double lower_bound = step_reward;
	double upper_bound = step_reward;

	// Create new belief nodes
  // For an observation, combine all the particles into 1 new VNode.
  // This vnode is the child of the qnode.
  int counter = 0;
	for (map<OBS_TYPE, vector<State*> >::iterator it = partitions.begin();
      it != partitions.end(); it++)
  {
		OBS_TYPE obs = it->first;
		// logd << " Creating node for obs " << obs << endl;

    // This vnode has all the partitions under obs.
    // Its parent is qnode, the edge between the qnode and the vnode
    // is the observation obs.
		VNode* vnode = new VNode(
        partitions[obs], parent->depth() + 1, qnode, obs);
		// logd << " New node created!" << endl;
		children[obs] = vnode;

    cout << "partitions[" << counter << "]: obs=" << obs
      << ", action=" << qnode->edge() << ", "
      << it->second.size() << " particles=" << endl;
    counter ++;
    // for (State* s: it->second)
    // {
      // cout << *s << endl;
    // }
    // TIGER:
    // partitions[0]: obs=2, action=0, 500 particles=
    // Since the action0=LEFT, we always get the observation=2.
    // So there is only 1 element in partitions.

    // We just create a new vnode for the action and observation
    // as a child of the qnode. Now we will compute the bounods of the vnode.
		history.Add(qnode->edge(), obs);

    // XXX: Why using bound?
    // Here we won't know the exact value of this new VNode vnode.
    // If we want to compute the exact q-value of qnode,
    // then we have to know the exact value of vnode. 
    // To compute the exact value of vnode, we need to keep growing the tree
    // to very deep until the exact value of vnode converges.
    // It is computationally heavy!
    //
    // Therefore, instead of aiming at the exact value of vnode,
    // we use the lower and upper bounds of vnode as an approximation.
    // In this way, we won't need to grow the tree till the exact value
    // of vnode converges.
    // We will keep growing the tree till the Weighted Excess
    // Utility (or the gap between the lower and upper bound) of the vnode
    // converges.

    // use the object lb and ub to update the bounds inside vnode.
		InitBounds(vnode, lb, ub, streams, history);
		history.RemoveLast();

    logd << " New node's bounds: (" << vnode->lower_bound() << ", "
      << vnode->upper_bound() << ")" << endl;
    // TIGER:
    // [DESPOT::InitBounds()]
    //
    // [DESPOT::InitLowerBound()]
    // Old vnode=@VNode:
    // [depth_=1, lower_bound_=1.4937e-316, upper_bound_=1.49371e-316]@
    // At depth=1, lower_bound (action, value) =(2, -20)
    // After discounted, (action, value)=(2, -19)
    // New vnode=@VNode:
    // [depth_=1, lower_bound_=-19, upper_bound_=1.49371e-316]@
    //
    // [DESPOT::InitUpperBound()]
    // Old vnode=@VNode:
    // [depth_=1, lower_bound_=-19, upper_bound_=1.49371e-316]@
    // At depth=1, upper_bound value =200
    // After discounted, value=190
    //
    // New vnode=@VNode: [depth_=1, lower_bound_=-19, upper_bound_=190]@
    // DEBUG:  New node's bounds: (-19, 190)

		lower_bound += vnode->lower_bound();
		upper_bound += vnode->upper_bound();
	}
	qnode->step_reward = step_reward;
	qnode->lower_bound(lower_bound);
	qnode->upper_bound(upper_bound);
	qnode->utility_upper_bound = upper_bound + Globals::config.pruning_constant;

	qnode->default_value = lower_bound; // for debugging

  cout << "After creating new vnodes as children, qnode=" << *qnode << endl;
  // TIGER:
  // After creating new vnodes as children, qnode=#QNode:
  // [lower_bound_=-62.24, upper_bound_=145.66, step_reward=-43.24,
  // utility_upper_bound=146.76]#
  // qnode.lower_bound_ = -19 + -43.24
  // qnode.upper_bound_ = -190 + -43.24
}

ValuedAction DESPOT::Evaluate(VNode* root, vector<State*>& particles,
	RandomStreams& streams, POMCPPrior* prior, const DSPOMDP* model) {
	double value = 0;

	for (int i = 0; i < particles.size(); i++) {
		particles[i]->scenario_id = i;
	}

	for (int i = 0; i < particles.size(); i++) {
		State* particle = particles[i];
		VNode* cur = root;
		State* copy = model->Copy(particle);
		double discount = 1.0;
		double val = 0;
		int steps = 0;

		while (!streams.Exhausted()) {
			int action =
				(cur != NULL) ?
					OptimalAction(cur).action : prior->GetAction(*copy);

			assert(action != -1);

			double reward;
			OBS_TYPE obs;
			bool terminal = model->Step(*copy, streams.Entry(copy->scenario_id),
				action, reward, obs);

			val += discount * reward;
			discount *= Globals::Discount();

			if (!terminal) {
				prior->Add(action, obs);
				streams.Advance();
				steps++;

				if (cur != NULL && !cur->IsLeaf()) {
					QNode* qnode = cur->Child(action);
					map<OBS_TYPE, VNode*>& vnodes = qnode->children();
					cur = vnodes.find(obs) != vnodes.end() ? vnodes[obs] : NULL;
				}
			} else {
				break;
			}
		}

		for (int i = 0; i < steps; i++) {
			streams.Back();
			prior->PopLast();
		}

		model->Free(copy);

		value += val;
	}

	return ValuedAction(OptimalAction(root).action, value / particles.size());
}

void DESPOT::belief(Belief* b) {
	logi << "[DESPOT::belief] Start: Set initial belief." << endl;
	belief_ = b;
	history_.Truncate(0);

	lower_bound_->belief(b); // needed for POMCPScenarioLowerBound
	logi << "[DESPOT::belief] End: Set initial belief." << endl;
}

void DESPOT::Update(int action, OBS_TYPE obs)
{
  cout << "[DESPOT::Update]" << endl;

	double start = get_time_second();

  // Update the history and particles inside belief_ (reweigh and resample)
	belief_->Update(action, obs);
	history_.Add(action, obs);

	lower_bound_->belief(belief_);

	logi << "[Solver::Update] Updated belief, history and root with action "
		<< action << ", observation " << obs
		<< " in " << (get_time_second() - start) << "s" << endl;
}

} // namespace despot
