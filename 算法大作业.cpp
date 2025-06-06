#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <cstring>
using namespace std;

const int INF = 1e9;

struct Edge {
    int to, rev;
    int cap;
};

class MaxFlow {
public:
    MaxFlow(int n) : G(n), level(n), iter(n), original(n) {}

    void addEdge(int from, int to, int cap) {
        G[from].push_back({ to, (int)G[to].size(), cap });
        G[to].push_back({ from, (int)G[from].size() - 1, 0 });

        // 记录原始容量用于后续判断是否选中
        original[from].push_back({ to, cap });
    }

    int dinic(int s, int t) {
        int flow = 0;
        while (bfs(s, t)) {
            fill(iter.begin(), iter.end(), 0);
            int f;
            while ((f = dfs(s, t, INF)) > 0) {
                flow += f;
            }
        }
        return flow;
    }

    // 返回匹配关系（学生编号 -> 时间段编号），未匹配返回0
    vector<int> get_matches(int num_students, int num_slots) {
        vector<int> match(num_students + 1, 0); // 1-based
        for (int student = 1; student <= num_students; ++student) {
            for (const Edge& e : G[student]) {
                int to = e.to;
                if (to >= num_students + 1 && to <= num_students + num_slots) {
                    if (e.cap == 0) { // 说明该边满流（被使用）
                        match[student] = to - num_students; // 恢复为时间段编号
                        break;
                    }
                }
            }
        }
        return match;
    }

private:
    vector<vector<Edge>> G;
    vector<vector<pair<int, int>>> original; // 原始边和容量
    vector<int> level, iter;

    bool bfs(int s, int t) {
        fill(level.begin(), level.end(), -1);
        queue<int> q;
        level[s] = 0;
        q.push(s);
        while (!q.empty()) {
            int v = q.front(); q.pop();
            for (auto& e : G[v]) {
                if (e.cap > 0 && level[e.to] < 0) {
                    level[e.to] = level[v] + 1;
                    q.push(e.to);
                }
            }
        }
        return level[t] != -1;
    }

    int dfs(int v, int t, int f) {
        if (v == t) return f;
        for (int& i = iter[v]; i < G[v].size(); ++i) {
            Edge& e = G[v][i];
            if (e.cap > 0 && level[v] < level[e.to]) {
                int d = dfs(e.to, t, min(f, e.cap));
                if (d > 0) {
                    e.cap -= d;
                    G[e.to][e.rev].cap += d;
                    return d;
                }
            }
        }
        return 0;
    }
};

int main() {
    int num_students, num_slots, pile_per_slot;
    cout << "请输入学生人数：";
    cin >> num_students;
    cout << "请输入时间段数量：";
    cin >> num_slots;
    cout << "请输入每个时间段的充电桩数量：";
    cin >> pile_per_slot;

    int S = 0;
    int T = num_students + num_slots + 1;
    int total_nodes = T + 1;

    MaxFlow mf(total_nodes);

    // 源点 -> 学生
    for (int i = 1; i <= num_students; ++i) {
        mf.addEdge(S, i, 1);
    }

    // 输入每位学生可用时间段
    vector<vector<int>> student_slots(num_students + 1); // 1-based
    for (int i = 1; i <= num_students; ++i) {
        int cnt;
        cout << "请输入学生 " << i << " 可用时间段数量：";
        cin >> cnt;
        cout << "请输入时间段编号（范围1~" << num_slots << "）：";
        for (int j = 0; j < cnt; ++j) {
            int slot;
            cin >> slot;
            student_slots[i].push_back(slot);
        }
    }

    // 学生 -> 可选时间段
    for (int i = 1; i <= num_students; ++i) {
        for (int slot : student_slots[i]) {
            int slot_node = num_students + slot;
            mf.addEdge(i, slot_node, 1);
        }
    }

    // 时间段 -> 汇点
    for (int slot = 1; slot <= num_slots; ++slot) {
        int slot_node = num_students + slot;
        mf.addEdge(slot_node, T, pile_per_slot);
    }

    // 执行最大流匹配
    int max_match = mf.dinic(S, T);
    cout << "最多可以安排 " << max_match << " 位学生成功充电。" << endl;

    // 输出匹配结果
    vector<int> matches = mf.get_matches(num_students, num_slots);
    for (int i = 1; i <= num_students; ++i) {
        if (matches[i])
            cout << "学生 " << i << " 被安排在时间段 " << matches[i] << " 进行充电。" << endl;
        else
            cout << "学生 " << i << " 未被安排充电。" << endl;
    }

    return 0;
}
