from AIplayer2 import AIPlayer as AIPlayer2
from AIplayer1 import AIPlayer as AIPlayer1
from policy_value_net import PolicyValueNet
from AIplayer3 import AIPlayerplus    
# 导入黑白棋文件
from game import Game

# 人类玩家黑棋初始化
black_player =  AIPlayer2("X")

# AI 玩家 白棋初始化
white_player = AIPlayer1("O")

policy_value_net = PolicyValueNet(model_file='D:\文件\CS\AI\Ex2\\best_policy.model')

#white_player = AIPlayerplus(policy_value_net.policy_value_fn,1000)

# 游戏初始化，第一个玩家是黑棋，第二个玩家是白棋
game = Game(black_player, white_player)
game.run()
