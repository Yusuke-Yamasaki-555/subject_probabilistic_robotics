#!/usr/bin/env python3

import matplotlib as plt
import numpy as np
import pprint as pp

class Q_Learning():
  def __init__(self):
    """ 環境の定義（座標と報酬）

    1次元の数直線上を左右に動くロボット

          0      1     2     3     4     5     6
    0 [ -100  , -1  ,  0  , -1  , -1  , -1  , 100 ]

    start: [0, 2]
    goal: [0, 6]

    """
    # 環境のサイズ
    self.env_size_ :tuple[int, int] = (0, 6)

    # 開始位置（＝初期位置）の座標（タプル）
    self.start_pos_ = np.array([self.env_size_[0], 2])
    # ゴール位置の座標（タプル）
    self.goal_pos_ = np.array([self.env_size_[0], self.env_size_[1]])

    # alfaの定義
    self.alfa = 0.5

    # epsilonの定義
    self.epsilon = 0.2

    # 報酬の変数（１行行列）
    self.env_reward_ = np.array([[-100, -1, 0, -1, -1, -1, 10]])
    self.env_reward_goal_ = self.env_reward_[self.goal_pos_[0]][self.goal_pos_[1]]
    self.reward_total_ = 0
    self.reward_ep_list_ :list[list[int]] = [[]]
    self.reward_total_ep_list_ :list[int] = []

    # Q値の変数（１行行列）
    self.q_ = np.full((self.env_size_[0]+1, self.env_size_[1]+1), 0)
    # 状態の変数（＝現在座標の変数。タプル）？
    self.x_ :np.ndarray = self.start_pos_.copy()
    # 行動の変数（＝速度の変数（X方向に+-1 のみ）。タプル）
    self.a_ = np.array([0, 0])  # [y, x]

    # 1step後の状態
    # self.x_next_ = np.array([None, None])

    # 状態と行動のlist
    self.x_ep_list_ :list[list[int]] = [[]]
    self.a_ep_list_ :list[list[int]] = [[]]
    
    # ゴール、学習の終了の判定を記録する変数
    self.check_goal_ = False
    self.check_rl_end_ = False

    # 現在および最大episode数の変数
    self.episode_max_ = 1000
    self.episode_now_ = 1

    # step数とepisode毎step数の変化の変数
    self.step_total_ = 1
    self.step_ep_list_ :list[int] = []


  def ResetVariable(self):
    # 変数を初期化

    self.reward_total_ = 0
    self.x_ :np.ndarray = self.start_pos_.copy()
    self.a_ = np.array([0, 0])
    self.x_ep_list_.append([])
    self.a_ep_list_.append([])
    self.reward_ep_list_.append([])
    # self.x_next_ = np.array([None, None])
    self.check_goal_ = False
    self.step_total_ = 1
    

  def RestartEpisode(self):
    # ゴールにたどり着いた場合に実行
      # Q値の行列以外の変数を初期値にリセット

    self.reward_total_ep_list_.append(self.reward_total_)
    self.step_ep_list_.append(self.step_total_)

    self.ResetVariable()

    if self.episode_now_ == self.episode_max_:
      self.check_rl_end_ = True
    # self.check_rl_end_ = True

    
  def Step(self):
    # 毎Stepの処理

    # a_tの決定からq_nowの決定
    if self.x_[1] == 0:
      self.a_ = np.array([0, 1])
    elif self.x_[1] == self.env_size_[1]:
      self.a_ = np.array([0, -1])
    else:
      if np.random.rand(1) < self.epsilon:
        self.a_ = np.array([0, np.random.choice((-1, 1))])
      else:
        if self.q_[self.x_[0]][self.x_[1]+1] > self.q_[self.x_[0]][self.x_[1]-1]:
          self.a_ = np.array([0, 1])
        elif self.q_[self.x_[0]][self.x_[1]+1] < self.q_[self.x_[0]][self.x_[1]-1]:
          self.a_ = np.array([0, -1])
        else:
          self.a_ = np.array([0, np.random.choice((-1, 1))])
    x_next = self.x_ + self.a_
    q_now = self.q_[x_next[0]][x_next[1]]

    # a_t+1の決定からq_nextの決定
    a_next = 0
    q_next = 0
    if x_next[1] == 0:
      a_next = np.array([0, 1])
    elif x_next[1] == self.env_size_[1]:
      a_next = np.array([0, -1])
    else:
      if self.q_[x_next[0]][x_next[1]+1] > self.q_[x_next[0]][x_next[1]-1]:
        a_next = np.array([0, 1])
      elif self.q_[x_next[0]][x_next[1]+1] < self.q_[x_next[0]][x_next[1]-1]:
        a_next = np.array([0, -1])
      else:
        a_next = np.array([0, np.random.choice((-1, 1))])
    
    x_next_next = x_next + a_next
    q_next = self.q_[x_next_next[0]][x_next_next[1]]

    r = self.env_reward_[self.x_[0]][self.x_[1]]
    self.reward_total_ += r
    q_now = (1 - self.alfa) * q_now + self.alfa * (r + q_next)

    self.q_[self.x_[0]][self.x_[1]] = q_now
    self.x_ep_list_[self.episode_now_-1].append(self.x_.tolist())
    self.a_ep_list_[self.episode_now_-1].append(self.a_.tolist())
    self.reward_ep_list_[self.episode_now_-1].append(r)
    if self.x_.tolist() == self.goal_pos_.tolist():
      self.check_goal_ = True
    else:
      self.x_ += self.a_


  def VisualizationResult(self):
    # 学習結果を升目上のマップの画像として可視化
      # 水たまりと普通で色分け
      # 最適経路で色分け
      # 各マスに各状態におけるQ値の最大値を表示

    pp.pprint(self.x_ep_list_)
    pp.pprint(self.a_ep_list_)
    pp.pprint(self.reward_ep_list_)
    pp.pprint(self.q_)
    print(self.reward_total_ep_list_)
    print(self.step_ep_list_)
    
    



if __name__ == "__main__":
  rl = Q_Learning()

  while True:

    rl.Step()

    if rl.check_goal_ == True:
      rl.RestartEpisode()

      if rl.check_rl_end_ == True:
        rl.VisualizationResult()
        break

      rl.episode_now_ += 1
      continue

    rl.step_total_ += 1