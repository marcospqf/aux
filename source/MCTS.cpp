#include "MCTS.hpp"
const int kINF = 0x3f3f3f3f;
double MCTS::Cp = 2.0 / sqrt(2);
bool MCTS::shutdown = false;
const int LOWER_BOUND = 12;
Graph::Graph(int n) : nvertex(n), graph(n + 1) {}
Graph::Graph(int n, vector<vector<int>> graph_) : nvertex(n), graph(graph_) {}

void Graph::AddEdge(int u, int v) {
  graph[u].push_back(v);
  graph[v].push_back(u);
}

set<int> Graph::NextCandidates(int vertex, set<int> &candidates) {
  set<int> nxt_candidates;
  for (int v : graph[vertex])
    if (candidates.count(v))
      nxt_candidates.insert(v);

  return nxt_candidates;
}

set<int> Graph::MaxCliqueMaxDegreeHeuristic(set<int> candidates,
                                            set<int> clique) {

  int len = candidates.size();
  vector<set<int>> graph_aux(len);
  vector<int> degree(len, 1);
  vector<int> vertex(len, 0);
  map<int, int> vertex_to_pos;
  int cnt = 0;
  for (int u : candidates) {
    vertex[cnt] = u;
    vertex_to_pos[u] = cnt++;
  }
  for (int u : candidates)
    for (int v : graph[u])
      if (candidates.count(v)) {
        graph_aux[vertex_to_pos[u]].insert(vertex_to_pos[v]);
        graph_aux[vertex_to_pos[v]].insert(vertex_to_pos[u]);
      }
  for (int i = 0; i < len; i++)
    degree[i] = graph_aux[i].size() + 1;

  for (int i = 0; i < len; i++) {
    int maxi = 0;
    int u = 0;
    for (int j = 0; j < len; j++) {
      if (degree[j] > maxi and degree[j] > 0)
        maxi = degree[j], u = j;
    }
    if (maxi == 0)
      break;
    clique.insert(vertex[u]);
    degree[u] = 0;
    for (int i = 0; i < len; i++)
      if (!graph_aux[u].count(i))
        degree[i] = 0;
  }
  return clique;
}
set<int> Graph::MaxCliqueMinDegreeHeuristic(set<int> candidates,
                                            set<int> clique) {

  int len = candidates.size();
  vector<set<int>> graph_aux(len);
  vector<int> degree(len, 1);
  vector<int> vertex(len, 0);
  map<int, int> vertex_to_pos;
  int cnt = 0;
  for (int u : candidates) {
    vertex[cnt] = u;
    vertex_to_pos[u] = cnt++;
  }

  for (int u : candidates)
    for (int v : graph[u])
      if (candidates.count(v)) {
        graph_aux[vertex_to_pos[v]].insert(vertex_to_pos[u]);
        graph_aux[vertex_to_pos[u]].insert(vertex_to_pos[v]);
      }

  for (int i = 0; i < len; i++) {
    int mini = kINF;
    int u = 0;
    for (int j = 0; j < len; j++) {
      if (degree[j] < mini and degree[j] > 0)
        mini = degree[j], u = j;
    }
    if (mini == kINF)
      break;
    clique.insert(vertex[u]);
    degree[u] = 0;
    for (int i = 0; i < len; i++)
      if (!graph_aux[u].count(i))
        degree[i] = 0;
  }

  return clique;
}

set<int> Graph::MaxCliqueRandomHeuristic(set<int> candidates, set<int> clique) {
  vector<int> cand;
  for (int x : candidates)
    cand.push_back(x);
  random_shuffle(cand.begin(), cand.end());

  for (int u : cand) {
    int nvis = 0;
    for (int v : graph[u])
      nvis += clique.count(v);
    if (nvis == clique.size())
      clique.insert(u);
  }
  return clique;
}

set<int> Graph::Solver(set<int> candidates, set<int> clique) {
  map<int, int> map_to_idx;
  map<int, int> trad;
  int n = 0;
  for (int u : candidates) {
    trad[n] = u;
    map_to_idx[u] = n++;
  }
  vector<long long> g(n, 0);
  vector<long long> dp((1ll << (n / 2 + 2)), 0);
  vector<long long> dp2((1ll << (n / 2 + 2)), 0);
  for (int u : candidates) {
    g[map_to_idx[u]] |= (1ll << map_to_idx[u]);
    for (int v : graph[u])
      if (candidates.count(v)) {
        g[map_to_idx[u]] |= (1ll << map_to_idx[v]);
      }
  }
  long long t1 = n / 2;
  long long t2 = n - t1;
  long long r = 0;
  long long maximum_clique = 0;
  for (long long mask = 1; mask < (1ll << t1); mask++) {
    for (long long j = 0; j < t1; j++)
      if (mask & (1ll << j)) {
        long long outra = mask - (1ll << j);
        long long r1 = __builtin_popcountll(dp[mask]);
        long long r2 = __builtin_popcountll(dp[outra]);
        if (r2 > r1)
          dp[mask] = dp[outra];
      }
    bool click = true;
    for (long long j = 0; j < t1; j++)
      if ((1ll << j) & mask)
        if (((g[j] ^ mask) & mask))
          click = false;
    if (click)
      dp[mask] = mask;
    long long r1 = __builtin_popcountll(dp[mask]);
    // r = max(r, r1);
    if (r1 > r)
      r = r1, maximum_clique = dp[mask];
  }

  for (long long mask = 1; mask < (1ll << t2); mask++) {
    for (long long j = 0; j < t2; j++)
      if (mask & (1ll << j)) {
        long long outra = mask - (1ll << j);
        long long r1 = __builtin_popcountll(dp2[mask]);
        long long r2 = __builtin_popcountll(dp2[outra]);
        if (r2 > r1)
          dp2[mask] = dp2[outra];
      }
    bool click = true;
    for (long long j = 0; j < t2; j++) {
      if ((1ll << j) & mask) {
        long long m1 = g[j + t1];
        long long cara = mask << t1;
        if ((m1 ^ cara) & cara) {
          click = false;
        }
      }
    }
    if (click) {
      dp2[mask] = mask;
    }
    long long r1 = __builtin_popcountll(dp2[mask]);
    if (r1 > r)
      r = r1, maximum_clique = dp2[mask];
    r = max(r, r1);
  }

  for (long long mask = 0; mask < (1ll << t1); mask++) {
    long long tudo = (1ll << n) - 1;
    for (long long j = 0; j < t1; j++)
      if ((1ll << j) & mask)
        tudo &= g[j];

    tudo >>= t1;
    long long x = __builtin_popcountll(dp[mask]);
    long long y = __builtin_popcountll(dp2[tudo]);
    if (x + y > r) {
      r = x + y;
      maximum_clique = dp[mask];
      for (long long j = 0; j < t2; j++)
        if ((1ll << j) & dp2[tudo])
          maximum_clique |= (1ll << (t1 + j));
    }
    r = max(r, x + y);
  }
  set<int> maxi_clique;
  for (int i = 0; i < n; i++)
    if ((1ll << i) & maximum_clique)
      maxi_clique.insert(trad[i]);
  assert(maxi_clique.size() == r);
  for (int x : maxi_clique) {
    assert(!clique.count(x));
    clique.insert(x);
  }
  return clique;
}

int Graph::UpperBoundClique(set<int> &clique, set<int> &candidates) {
  return clique.size() + candidates.size();
}

/*-----------------------*/
State::State(set<int> clique_, vector<State *> son_, set<int> candidates_,
             int nvisited_, double sum_reward_, bool is_terminal_,
             int upper_bound_clique_)
    : clique(clique_), son(son_), nvisited(nvisited_), candidates(candidates_),
      sum_reward(sum_reward_), is_terminal(is_terminal_),
      upper_bound_clique(upper_bound_clique_) {}

double State::GetReward(int nvis, double normalize) {
  if (nvisited == 0)
    return kINF;
  return sum_reward / (nvisited * normalize) +
         MCTS::Cp * sqrt((2.0 * log(nvis)) / (double)nvisited);
}
int State::GetBestChild() {
  assert(nvisited > 0);
  double maxi = -kINF;
  vector<int> best;
  double maxi_mean = -kINF;
  vector<int> idx_terminals;
  assert(sum_reward > 0);
  for (int i = 0; i < (int)son.size(); i++) {
    if (son[i] != nullptr) {
      if (son[i]->is_terminal) {
        idx_terminals.push_back(i);
      } else {
        maxi_mean = max(maxi_mean, son[i]->sum_reward / nvisited);
      }
    }
  }
  if (maxi_mean < 1)
    maxi_mean = 1;
  if (idx_terminals.size() > 0) {
    int tam = idx_terminals.size();
    int vai = rand() % tam;
    return idx_terminals[vai];
  }
  int ok = 0;
  //  std::cout << maxi_mean << endl;
  for (int i = 0; i < (int)son.size(); i++) {
    if (son[i] != nullptr) {
      ok = 1;
      double uct = son[i]->GetReward(nvisited, maxi_mean);
      //   std::cout << uct << endl;
      if (uct > maxi and fabs(uct - maxi) > 1e-8) {
        best.clear();
        maxi = uct;
        best.push_back(i);
      } else if (fabs(uct - maxi) <= 1e-8) {
        best.push_back(i);
      }
    }
  }
  if (best.size() == 0) {
    assert(!ok);
    return -1;
  }
  int tam = best.size();
  return best[rand() % tam];
}

/*--------------------------*/

MCTS::MCTS(int n, vector<vector<int>> graph_) {
  graph = new Graph(n, graph_);
  set<int> candidates;
  for (int i = 1; i <= n; i++)
    candidates.insert(i);
  root = new State({}, {}, candidates, 0, 0, true, kINF);
  shutdown = false;
}

void MCTS::SetShutDown(int signum) { shutdown = true; }

set<int> MCTS::Process() {
  int cnt = 0;
  while (root != nullptr and !shutdown) {
    root = Build(root).first;
    if (cnt % 100 == 0)
      std::cout << maximum_clique.size() << endl;
    cnt++;
  }
  return maximum_clique;
}

vector<State *> MCTS::GenChildren(State *tree_vertex) {
  vector<State *> result;
  set<int> clique = tree_vertex->clique;
  set<int> candidates = tree_vertex->candidates;
  State *with_vertex = nullptr;
  State *without_vertex = nullptr;
  if (candidates.size() > 0) {
    int idx = rand() % candidates.size();
    assert(idx >= 0);
    int vertex = -1;
    for (int x : candidates)
      if (idx == 0)
        vertex = x;
      else
        idx--;
    assert(vertex != -1);
    assert(candidates.count(vertex));
    candidates.erase(vertex);
    without_vertex = new State(clique, {}, candidates, 0, 0, true,
                               graph->UpperBoundClique(clique, candidates));
    clique.insert(vertex);
    set<int> next_cand = graph->NextCandidates(vertex, candidates);
    with_vertex = new State(clique, {}, next_cand, 0, 0, true,
                            graph->UpperBoundClique(clique, next_cand));
    result.push_back(without_vertex);
    result.push_back(with_vertex);
  }
  return result;
}

State *MCTS::Expand(State *tree_vertex) {
  set<int> clique = tree_vertex->clique;
  set<int> candidates = tree_vertex->candidates;

  if (candidates.size() <= 30) {
    set<int> clique2 = graph->Solver(candidates, clique);
    if (maximum_clique.size() < clique2.size())
      maximum_clique = clique2;
    return nullptr;
  }

  vector<State *> son = GenChildren(tree_vertex);
  int nvisited = 1;
  bool is_terminal = false;
  set<int> simu_clique = Simulation(tree_vertex);
  if (maximum_clique.size() < simu_clique.size())
    maximum_clique = simu_clique;
  if (son.size() > 0) {
    assert(simu_clique.size() > 0);
    double sum_reward = simu_clique.size();
    return new State(clique, son, candidates, nvisited, sum_reward, is_terminal,
                     graph->UpperBoundClique(clique, candidates));
  }
  assert(son.size() == 0);
  return nullptr;
}
pair<State *, double> MCTS::Build(State *tree_vertex) {
  //  std::cout << "oi" << endl;
  tree_vertex->nvisited++;
  if (tree_vertex->is_terminal) {
    tree_vertex = Expand(tree_vertex);
    if (tree_vertex == nullptr) {
      return {nullptr, 0};
    }
    assert(!tree_vertex->is_terminal);
    if (tree_vertex->upper_bound_clique <= (int)maximum_clique.size()) {
      EraseBranch(tree_vertex);
      return {nullptr, 0};
    }
    return {tree_vertex, tree_vertex->sum_reward};
  }
  int idx = tree_vertex->GetBestChild();
  if (idx == -1) {
    tree_vertex->son.clear();
    delete tree_vertex;
    tree_vertex = nullptr;
    return {nullptr, 0};
  }
  pair<State *, double> new_child = Build(tree_vertex->son[idx]);
  tree_vertex->son[idx] = new_child.first;
  tree_vertex->sum_reward += new_child.second;
  if (tree_vertex->upper_bound_clique <= LOWER_BOUND) {
    EraseBranch(tree_vertex);
    return {nullptr, 0};
  }

  if (tree_vertex->upper_bound_clique <= (int)maximum_clique.size()) {
    EraseBranch(tree_vertex);
    return {nullptr, 0};
  }
  return {tree_vertex, new_child.second};
}

set<int> MCTS::Simulation(State *tree_vertex) {
  set<int> candidates = tree_vertex->candidates;
  set<int> c3 = graph->MaxCliqueRandomHeuristic(tree_vertex->candidates,
                                                tree_vertex->clique);

  if (candidates.size() <= 500) {
    //    std::cout << "CHAMOU" << endl;
    set<int> c1 = graph->MaxCliqueMinDegreeHeuristic(tree_vertex->candidates,
                                                     tree_vertex->clique);
    //    std::cout << tree_vertex->clique.size() << " " << c1.size() << endl;
    if (c3.size() < c1.size())
      c3 = c1;
    set<int> c2 = graph->MaxCliqueMaxDegreeHeuristic(tree_vertex->candidates,
                                                     tree_vertex->clique);
    if (c3.size() < c2.size())
      c3 = c2;
  }
  return c3;
}

void MCTS::EraseBranch(State *tree_vertex) {
  if (tree_vertex == nullptr)
    return;
  for (auto son : tree_vertex->son)
    EraseBranch(son);

  tree_vertex->son.clear();
  delete tree_vertex;
  tree_vertex = nullptr;
}
