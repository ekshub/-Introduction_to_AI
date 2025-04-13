import math
import copy
import numpy as np
import random
from board import Board

'''
此为采用神经网络改进探索策略的蒙特卡洛树搜索
未考虑蒙特卡洛树搜索中对方或者我方连续下棋
'''

def softmax(x):
    probs = x
    probs /= np.sum(probs)
    return probs


class Node_plus(object):
    '''
    建立节点。每个节点存储当前棋盘状态、走法候选、统计数据以及神经网络返回的先验概率。
    '''
    def __init__(self):
        self.color = None
        self.board = Board()   # 保存当前棋局状态
        self.candidate = None  # 该节点对应的走法（格式与 board.get_legal_actions 返回值一致，例如 "D3" ）
        self.visit = 0
        self.score = 0       # 记录累计得分（黑棋赢为 +1 分，白棋赢为 -1 分，平局为 0）
        self.parent = None
        self.child = []      # 从该节点展开的子节点（Node_plus 对象）
        self.childnodes = [] # 子节点对应的走法坐标列表
        # 使用 get_legal_actions 替代 locations 方法
        self.next_locations = None  
        self.status = 0      # 0为未完全扩展，1为完全扩展
        self.prob = 1        # 该节点的先验概率（由策略网络提供）
        self.nextlocation_prob = None  # 对应各候选走法的先验概率，格式为8x8数组或字典，具体由网络返回


class Mcts_plus(object):
    '''
    蒙特卡洛树搜索实现，结合神经网络改进了探索策略。
    参数说明：
      board: 当前棋局状态，必须含有 color 属性（例如 "X" 或 "O"）
      policy_value_function: 神经网络接口，输入 board，返回 (nextlocation_prob, score)
      r: 搜索迭代次数
      is_selfplay: 是否自对弈（1 为自对弈模式，否则为对战模式）
    '''
    
    def __init__(self, board, policy_value_function, r, is_selfplay=0):
        self.color = board.color 
        self.board = copy.deepcopy(board)
        self.r = r    # 迭代次数
        self.func = policy_value_function
        self.is_selfplay = is_selfplay
        
    def ucb1(self, node, c=1/math.sqrt(2)): 
        '''
        UCB1计算公式：
          l = (score/visit) + c * prior * sqrt(2 * parent.visit) / (1 + visit)
        '''
        l = node.score / node.visit + c * node.prob * math.sqrt(2 * node.parent.visit) / (1 + node.visit)
        return l
        
    def selection(self, node):
        '''
        从当前节点开始，依据 UCB1 值选择到叶子节点
        '''
        selection_node = node
        while selection_node.status == 1 and selection_node.child:
            best_child_value = -float('inf')
            best_child = None
            for child in selection_node.child:
                ucb_val = self.ucb1(child)
                if ucb_val > best_child_value:
                    best_child_value = ucb_val
                    best_child = child
            selection_node = best_child
        return selection_node
         
    def expand(self, node):
        '''
        对节点进行扩展。使用策略网络返回的先验概率选择未扩展的候选走法。
        '''
        # 获取当前节点所有合法走法（转为列表）
        if node.next_locations is None:
            node.next_locations = list(node.board.get_legal_actions(node.board.color))
        
        no_child_candidates = []
        for cand in node.next_locations:
            if cand not in node.childnodes:
                no_child_candidates.append(cand)
                
        if len(no_child_candidates) == 1:
            node.status = 1  # 标记为完全扩展
        
        max_candidate_prob = -float('inf')
        for cand in no_child_candidates:
            # 假设 nextlocation_prob 为8x8数组，下标对应棋盘坐标（行, 列）
            # 需要将走法 cand 转换为数字坐标；board.py中通常采用 board.board_num(cand)
            # 这里假设 cand 格式可以直接映射为 (row, col) 如 "D3" 由 board.board_num 转换为 (2,3)
            x, y = node.board.board_num(cand)
            prob = node.nextlocation_prob[x][y]
            if prob > max_candidate_prob:
                max_candidate_prob = prob
                expand_node_candidate = cand

        node.childnodes.append(expand_node_candidate)
        expand_node = Node_plus()
        # 根节点扩展保留原始颜色，否则切换颜色
        if node.parent is None:
            expand_node.color = self.color
        else:
            expand_node.color = 'O' if node.color == 'X' else 'X'
        expand_node.prob = max_candidate_prob
        expand_node.parent = node
        expand_node.candidate = expand_node_candidate
        
        node.child.append(expand_node)
        return expand_node
       
    def simulation(self, node):
        '''
        通过神经网络评估当前局面，不进行随机完整模拟。
        这里调用策略价值网络，并反转返回的 score（因为网络返回的是对手局面评分）。
        '''
        board_copy = copy.deepcopy(node.board)
        # 获取合法走法（调用 get_legal_actions，将生成器转换为 list）
        board_copy.get_legal_actions(node.board.color)  # 可选：确保局面状态更新
        # 调用策略价值网络得到先验概率和局面评估
        node.nextlocation_prob, node.score = self.func(board_copy)
        node.score = -node.score  # 反转视角
       
    def back_update(self, node):
        '''
        从当前节点向上回溯，更新访问次数和累计得分
        '''
        score = node.score
        while node.parent is not None:
            node.parent.visit += 1
            score = -score  # 反转视角
            node.parent.score += score  
            node = node.parent

    def mcts_run(self):
        '''
        执行蒙特卡洛树搜索。
        对战模式下返回访问数最高的走法，自对弈模式下返回带噪声的概率分布。
        '''
        root = Node_plus()
        root.color = self.color 
        root.board = self.board
        root.board.color = root.color
        # 初始候选走法：调用 get_legal_actions，并转换为 list
        root.next_locations = list(root.board.get_legal_actions(root.board.color))
        self.simulation(root)
        
        # 第一次扩展（不进行选择，直接扩展）
        expand_node = self.expand(root)
        # 复制棋盘状态并落子。这里用 _move 代替原来的 reversi_pieces 方法
        board_copy = copy.deepcopy(self.board)
        board_copy._move(expand_node.candidate, board_copy.color)
        expand_node.board = board_copy
        expand_node.visit = 1
        
        # 切换局面颜色：棋盘颜色应为上一步棋对手颜色
        if expand_node.color == 'X':
            expand_node.board.color = 'O'
            self.simulation(expand_node)
        else:
            expand_node.board.color = 'X'
            self.simulation(expand_node)
            
        # 更新扩展节点的候选走法（list 格式）
        expand_node.next_locations = list(expand_node.board.get_legal_actions(expand_node.board.color))
        # 判断是否棋盘已满：这里使用 get_winner 进行判断
        if sum([expand_node.board.count(c) for c in ['X','O']]) == 64:
            winner, diff = expand_node.board.get_winner()
            if expand_node.color == 'X':
                expand_node.score = diff
            else:
                expand_node.score = -diff
        self.back_update(expand_node)
        
        i = 0
        while i < self.r and expand_node.next_locations:
            i += 1
            selection_node = self.selection(root)
            expand_node = self.expand(selection_node)
            board_copy2 = copy.deepcopy(selection_node.board)
            board_copy2._move(expand_node.candidate, board_copy2.color)
            expand_node.board = board_copy2
            expand_node.visit = 1
            
            if expand_node.color == 'X':
                expand_node.board.color = 'O'
                self.simulation(expand_node)
            else:
                expand_node.board.color = 'X'
                self.simulation(expand_node)
                
            expand_node.next_locations = list(expand_node.board.get_legal_actions(expand_node.board.color))
            if sum([expand_node.board.count(c) for c in ['X','O']]) == 64:
                winner, diff = expand_node.board.get_winner()
                if expand_node.color == 'X':
                    expand_node.score = diff
                else:
                    expand_node.score = -diff
            self.back_update(expand_node)
        
        action = None
        max_visit = 0
        mcts_visit = []
        mcts_prob = np.zeros((8, 8))
        if self.is_selfplay == 0:
            for node_child in root.child:
                if node_child.visit > max_visit:
                    max_visit = node_child.visit
                    action = node_child.candidate
                # 将候选走法转换为数字坐标用于 mcts_prob 记录
                a, b = root.board.board_num(node_child.candidate)
                mcts_prob[a][b] = node_child.visit
        else: 
            for node_child in root.child:
                a, b = root.board.board_num(node_child.candidate)
                mcts_prob[a][b] = node_child.visit
                mcts_visit.append(node_child.visit)
            mcts_visit = np.array(mcts_visit)
            # 使用 random.choices 按权重选取走法，此处加上 Dirichlet 噪声
            action_node = random.choices(
                root.child, 
                weights=0.75 * mcts_visit + 0.25 * np.random.dirichlet(0.3 * root.visit * np.ones(len(mcts_visit))),
                k=1
            )[0]
            action = action_node.candidate
        
        mcts_prob = softmax(mcts_prob)
        
        return action, mcts_prob
class AIPlayerplus():    # 利用结合神经网络的蒙特卡洛树搜索的AI玩家，迭代次数固定为100次
    '''
    超级电脑玩家
    '''
    def __init__(self, policy_value_function, mcts_n=400):
        self.mcts_n = mcts_n
        self.policy_value_function = policy_value_function
        
    def get_move(self, board):
        '''
        实际用 不传输mcts中数据
        '''
        board.pieces_index()
        
        action1 = Mcts_plus(board, self.policy_value_function, self.mcts_n).mcts_run()
        action = action1[0]
        return action
    
    def move1(self, board):
        '''
        自我对战用 需要传输数据
        '''
        board.pieces_index()

        action = Mcts_plus(board, self.policy_value_function, self.mcts_n, 1).mcts_run()
            
        return action