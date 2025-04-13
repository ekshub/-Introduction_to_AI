class Board(object):
    """
    Board 黑白棋棋盘，规格是8*8，黑棋用 X 表示，白棋用 O 表示，未落子时用 . 表示。
    """

    def __init__(self):
        """
        初始化棋盘状态
        """
        self.empty = '.'  # 未落子状态
        self._board = [[self.empty for _ in range(8)] for _ in range(8)]
        self._board[3][4] = 'X'
        self._board[4][3] = 'X'
        self._board[3][3], self._board[4][4] = 'O', 'O'
        # 初始化棋子计数
        self.pieces_index()  

    def __getitem__(self, index):
        """
        添加 Board[][] 索引语法
        """
        return self._board[index]

    def display(self, step_time=None, total_time=None):
        """
        打印棋盘
        """
        board = self._board
        print(' ', ' '.join(list('ABCDEFGH')))
        for i in range(8):
            print(str(i + 1), ' '.join(board[i]))
        if (not step_time) or (not total_time):
            step_time = {"X": 0, "O": 0}
            total_time = {"X": 0, "O": 0}
            print("统计棋局: 棋子总数 / 每一步耗时 / 总时间 ")
            print("黑   棋: " + str(self.count('X')) + ' / ' + str(step_time['X']) + ' / ' + str(total_time['X']))
            print("白   棋: " + str(self.count('O')) + ' / ' + str(step_time['O']) + ' / ' + str(total_time['O']) + '\n')
        else:
            print("统计棋局: 棋子总数 / 每一步耗时 / 总时间 ")
            print("黑   棋: " + str(self.count('X')) + ' / ' + str(step_time['X']) + ' / ' + str(total_time['X']))
            print("白   棋: " + str(self.count('O')) + ' / ' + str(step_time['O']) + ' / ' + str(total_time['O']) + '\n')

    def count(self, color):
        """
        统计 color 一方棋子的数量。(O:白棋, X:黑棋, .:未落子状态)
        """
        count = 0
        for y in range(8):
            for x in range(8):
                if self._board[x][y] == color:
                    count += 1
        return count

    def pieces_index(self):
        """
        更新棋盘上的棋子统计数据。
        将黑棋和白棋的数量分别保存到属性 black_count 和 white_count 中。
        """
        self.black_count = self.count('X')
        self.white_count = self.count('O')

    def get_winner(self):
        """
        通过棋子的个数判断胜负
        :return: 0-黑棋赢, 1-白棋赢, 2-平局, 同时返回胜负分差
        """
        black_count, white_count = 0, 0
        for i in range(8):
            for j in range(8):
                if self._board[i][j] == 'X':
                    black_count += 1
                if self._board[i][j] == 'O':
                    white_count += 1
        if black_count > white_count:
            return 0, black_count - white_count
        elif black_count < white_count:
            return 1, white_count - black_count
        else:
            return 2, 0

    def _move(self, action, color):
        """
        落子并返回翻转棋子的坐标列表
        """
        if isinstance(action, str):
            action = self.board_num(action)
        fliped = self._can_fliped(action, color)
        if fliped:
            for flip in fliped:
                x, y = self.board_num(flip)
                self._board[x][y] = color
            x, y = action
            self._board[x][y] = color
            # 更新棋子计数信息
            self.pieces_index()
            return fliped
        else:
            return False

    def backpropagation(self, action, flipped_pos, color):
        """
        回溯操作，撤销落子
        """
        if isinstance(action, str):
            action = self.board_num(action)
        self._board[action[0]][action[1]] = self.empty
        op_color = "O" if color == "X" else "X"
        for p in flipped_pos:
            if isinstance(p, str):
                p = self.board_num(p)
            self._board[p[0]][p[1]] = op_color
        self.pieces_index()

    def is_on_board(self, x, y):
        """
        判断坐标是否在棋盘范围内
        """
        return 0 <= x < 8 and 0 <= y < 8

    def _can_fliped(self, action, color):
        """
        判断落子是否合法，返回翻转子坐标列表或 False
        """
        if isinstance(action, str):
            action = self.board_num(action)
        xstart, ystart = action
        if not self.is_on_board(xstart, ystart) or self._board[xstart][ystart] != self.empty:
            return False
        self._board[xstart][ystart] = color
        op_color = "O" if color == "X" else "X"
        flipped_pos = []
        flipped_pos_board = []
        for xdirection, ydirection in [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]:
            x, y = xstart, ystart
            x += xdirection
            y += ydirection
            if self.is_on_board(x, y) and self._board[x][y] == op_color:
                x += xdirection
                y += ydirection
                if not self.is_on_board(x, y):
                    continue
                while self._board[x][y] == op_color:
                    x += xdirection
                    y += ydirection
                    if not self.is_on_board(x, y):
                        break
                if not self.is_on_board(x, y):
                    continue
                if self._board[x][y] == color:
                    while True:
                        x -= xdirection
                        y -= ydirection
                        if x == xstart and y == ystart:
                            break
                        flipped_pos.append([x, y])
        self._board[xstart][ystart] = self.empty  # 恢复空位
        if len(flipped_pos) == 0:
            return False
        for fp in flipped_pos:
            flipped_pos_board.append(self.num_board(fp))
        return flipped_pos_board

    def get_legal_actions(self, color):
        """
        根据棋规获取合法落子坐标
        """
        direction = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
        op_color = "O" if color == "X" else "X"
        op_color_near_points = []
        board = self._board
        for i in range(8):
            for j in range(8):
                if board[i][j] == op_color:
                    for dx, dy in direction:
                        x, y = i + dx, j + dy
                        if 0 <= x < 8 and 0 <= y < 8 and board[x][y] == self.empty and (x, y) not in op_color_near_points:
                            op_color_near_points.append((x, y))
        l = list(range(8))
        for p in op_color_near_points:
            if self._can_fliped(p, color):
                if p[0] in l and p[1] in l:
                    p = self.num_board(p)
                yield p

    def board_num(self, action):
        """
        棋盘坐标转数字坐标，例如 A1 --> (0,0)
        """
        row, col = str(action[1]).upper(), str(action[0]).upper()
        if row in '12345678' and col in 'ABCDEFGH':
            x, y = '12345678'.index(row), 'ABCDEFGH'.index(col)
            return x, y

    def num_board(self, action):
        """
        数字坐标转棋盘坐标，例如 (0,0) --> A1
        """
        row, col = action
        if col in range(8) and row in range(8):
            return chr(ord('A') + col) + str(row + 1)
