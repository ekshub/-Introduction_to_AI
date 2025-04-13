import math
import random
from copy import deepcopy
from func_timeout import func_timeout, FunctionTimedOut


class Node:
    
    
    
    # UCT 探索系数
    EXPLORATION_COEFFICIENT = 2

    def __init__(self, board, color, root_color, parent=None, pre_action=None):
        """
        初始化节点，保存棋盘状态、当前行棋方、合法动作等信息。
        """
        self.board = board
        self.color = color.upper()
        self.root_color = root_color.upper()
        self.parent = parent
        self.children = []           # 存储子节点列表
        self.pre_action = pre_action
        
        # 获取当前合法走法列表
        self.actions = list(self.board.get_legal_actions(color=self.color))
        # 判断当前局面是否结束
        self.is_over = self._check_game_over()

        # 初始化节点统计数据
        self.visit_count = 0
        self.reward = {'X': 0, 'O': 0}
        # 初始设为负无穷，未访问节点不参与最佳子节点比较
        self.value = {'X': float('-inf'), 'O': float('-inf')}
        self.is_leaf = True

         #初始化最佳子节点记录（根据 UCT 和奖励）
        self.best_child = None
        self.best_reward_child = None

    def _check_game_over(self):
        """
        判断游戏结束条件：当双方均无合法走法时，游戏结束。
        """
        return (len(list(self.board.get_legal_actions('X'))) == 0 and
                len(list(self.board.get_legal_actions('O'))) == 0)

    def update_value(self):
        """
        根据 UCT 公式更新当前节点的估值，计算平衡利用和探索的目标函数值。
        """
        if self.visit_count == 0 or self.parent is None:
            return
        for col in ['X', 'O']:
            exploitation = self.reward[col] / self.visit_count
            exploration = Node.EXPLORATION_COEFFICIENT * math.sqrt(
                2 * math.log(self.parent.visit_count) / self.visit_count
            )
            self.value[col] = exploitation + exploration

    def add_child(self, child):
        """
        添加新的子节点，并更新标记和最佳子节点记录。
        """
        self.children.append(child)
        self.is_leaf = False
        self.best_child = self._select_best_child()
        self.best_reward_child = self._select_best_reward_child()

    def _select_best_child(self):
        """
        按照当前玩家的 UCT 值选择最佳的子节点。
        """
        if not self.children:
            return None
        return max(self.children, key=lambda child: child.value.get(self.color, float('-inf')))

    def _select_best_reward_child(self):
        """
        根据模拟得到的奖励（胜率）选择最优子节点。
        """
        if not self.children:
            return None
        def reward_rate(child):
            if child.visit_count > 0:
                return child.reward[self.color] / child.visit_count
            return float('-inf')
        return max(self.children, key=reward_rate)
class MonteCarloSearch:
    def __init__(self, board, color, timeout=3):
        # 复制棋盘状态构造根节点
        self.root = Node(board=deepcopy(board), color=color, root_color=color)
        self.color = color.upper()
        self.timeout = timeout

        # 探索参数：初始 epsilon 及其衰减因子 gamma
        self.epsilon = 0.3
        self.gamma = 0.999

    def search(self):
        # 当根节点仅有一个合法动作时直接返回
        if len(self.root.actions) == 1:
            return self.root.actions[0]
        try:
            func_timeout(timeout=self.timeout, func=self._build_tree)
        except FunctionTimedOut:
            pass
        best_node = self.root._select_best_reward_child()
        return best_node.pre_action if best_node is not None else None

    def _build_tree(self):
        # 构建蒙特卡洛树，直至超时为止
        while True:
            current_node = self._select()
            # 终局判断
            if current_node.is_over:
                winner, diff = current_node.board.get_winner()
            else:
                # 对访问过的节点进行扩展
                if current_node.visit_count > 0:
                    current_node = self._expand(current_node)
                winner, diff = self._simulate(current_node)
            self._back_propagate(current_node, winner, diff)

    def _select(self):
        # 从根节点出发依据 epsilon-greedy 策略选择到叶子节点
        node = self.root
        current_epsilon = self.epsilon
        while not node.is_leaf:
            if random.random() > current_epsilon:
                child = node._select_best_child()
            else:
                child = random.choice(node.children)
            node = child
            current_epsilon *= self.gamma
        return node

    def _simulate(self, node):
        # 模拟从当前节点随机走子至游戏结束
        sim_board = deepcopy(node.board)
        sim_color = node.color
        while not self._is_game_over(sim_board):
            legal_actions = list(sim_board.get_legal_actions(color=sim_color))
            if legal_actions:
                sim_board._move(random.choice(legal_actions), sim_color)
            sim_color = 'X' if sim_color == 'O' else 'O'
        return sim_board.get_winner()

    def _expand(self, node):
        # 对当前节点所有合法走法生成子节点
        if not node.actions:
            new_board = deepcopy(node.board)
            next_color = 'X' if node.color == 'O' else 'O'
            child = Node(board=new_board, color=next_color, root_color=self.color,
                         parent=node, pre_action="none")
            node.add_child(child)
            return child

        for action in node.actions:
            new_board = deepcopy(node.board)
            new_board._move(action=action, color=node.color)
            next_color = 'X' if node.color == 'O' else 'O'
            child = Node(board=new_board, color=next_color, root_color=self.color,
                         parent=node, pre_action=action)
            node.add_child(child)
        return node._select_best_child()

    def _back_propagate(self, node, winner, diff):
        # 将模拟结果反向传播更新每个节点的统计数据
        while node is not None:
            node.visit_count += 1
            if winner == 0:
                node.reward['O'] -= diff
                node.reward['X'] += diff
            elif winner == 1:
                node.reward['X'] -= diff
            elif winner == 2:
                node.reward['O'] -= diff
            node.update_value()
            node = node.parent

    def _is_game_over(self, board):
        return (len(list(board.get_legal_actions('X'))) == 0 and
                len(list(board.get_legal_actions('O'))) == 0)
class AIPlayer:
    def __init__(self, color: str):
        self.color = color.upper()
        self.thinking_message = "请稍后，{}正在思考".format("黑棋(X)" if self.color == 'X' else "白棋(O)")

    def get_move(self, board):
        print(self.thinking_message)
        mcts = MonteCarloSearch(board, self.color)
        return mcts.search()