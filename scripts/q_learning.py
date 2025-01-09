#!/usr/bin/env python3

""" LICENSE

SPDX-License-Identifier:MIT
Copyright (C) 2024 Yusuke Yamasaki. All Rights Reserved.

"""


""" References

The GUIImage function included in this program was written with reference to the following website:

 - matplotlib, "Annotated heatmap", https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py, last viewed on: 2025/1/10

"""


import matplotlib.pyplot as plt
import numpy as np

class Q_Learning():
  def __init__(self):
    """ 環境の定義（座標と報酬）

    1次元の数直線上を左右に動くロボット

          0      1     2     3     4     5     6
    0 [ -100  , -1  ,  0  , -1  , -1  , -1  , 10  ]

    start: [0, 2]
    goal : [0, 6]

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
    # 状態の変数（＝現在座標の変数。タプル）
    self.x_ :np.ndarray = self.start_pos_.copy()
    # 行動の変数（＝速度の変数（X方向に+-1 のみ）。タプル）
    self.a_ = np.array([0, 0])  # [y, x]

    # 状態と行動のlist
    self.x_ep_list_ :list[list[int]] = [[]]
    self.a_ep_list_ :list[list[int]] = [[]]
    
    # ゴール、学習の終了の判定を記録する変数
    self.check_goal_ = False
    self.check_rl_end_ = False

    # 現在および最大episode数の変数
    self.episode_max_ = 100
    self.episode_now_ = 1

    # step数とepisode毎step数の変化の変数
    self.step_total_ = 1
    self.step_ep_list_ :list[int] = []

    # 最良なエピソードを記憶する変数
    self.best_episode_ = 0
    self.best_reward_ = -10000
    self.best_x_ :list[int] = []
    self.best_a_ :list[int] = []
    self.best_q_ :list[int] = []


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

    # ここに、最良の場合の累積報酬、Q値リスト、行動リスト＝方策を別変数にコピー記録
    if self.best_reward_ < self.reward_total_:
      self.best_episode_ = self.episode_now_
      self.best_reward_ = self.reward_total_
      self.best_x_ = self.x_ep_list_[-1]
      self.best_a_ = self.a_ep_list_[-1]
      self.best_q_ = self.q_.tolist()

    self.ResetVariable()

    if self.episode_now_ == self.episode_max_:
      self.check_rl_end_ = True

    
  def Step(self):
    # 毎Stepの処理

    # a_tの決定からq_nowの決定
    if self.x_[1] == 0:  # 停止できず左右にしか動けないので、環境の端にいる場合は移動方向を固定
      self.a_ = np.array([0, 1])
    elif self.x_[1] == self.env_size_[1]:
      self.a_ = np.array([0, -1])
    else:
      if np.random.rand(1) < self.epsilon:  # epsilon-greedy方策
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

    # Q値の更新
    r = self.env_reward_[self.x_[0]][self.x_[1]]
    self.reward_total_ += r
    q_now = (1 - self.alfa) * q_now + self.alfa * (r + q_next)

    # 次Step用の更新および記録
    self.q_[self.x_[0]][self.x_[1]] = q_now
    self.x_ep_list_[self.episode_now_-1].append(self.x_.tolist())
    self.a_ep_list_[self.episode_now_-1].append(self.a_.tolist())
    self.reward_ep_list_[self.episode_now_-1].append(r)
    if self.x_.tolist() == self.goal_pos_.tolist():
      self.check_goal_ = True
    else:
      self.x_ += self.a_

  def GUIImage(self):
    # 学習結果を升目上のマップの画像として可視化
      # 水たまりと普通で色分け
      # 最適経路で色分け
      # 各マスに各状態におけるQ値の最大値を表示

    """ References
      This function was written with reference to the following website:

       - matplotlib, "Annotated heatmap", https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py, last viewed on: 2025/1/10
    """

    _, axes = plt.subplots()
    axes.imshow(self.best_q_, cmap="PiYG")  # Gridの描画
    for i in range(len(self.best_q_[0])):  # テキストの描画
      axes.text(i, 0, self.best_q_[0][i], horizontalalignment="center", verticalalignment="center")
    axes.set_yticks(range(len([0])), "0")
    plt.title("Learning Results of Q Learning: The best Q values at each coordinate")
    plt.show()


  def VisualizationResult(self):
    # CLI上に学習結果を表示。GUIでの可視化を行う関数の呼び出し

    op_env = \
      """
----------

[ Q Learning ]

----------

Settings

- Environment: 
      |     0       1       2       3       4       5       6
  ----|------------------------------------------------------------
    0 | [ (0,0) , (0,1) , (0,2) , (0,3) , (0,4) , (0,5) , (0,6) ]
            |               |                               |  
          <Puddle>        <Start>                          <Goal>
                            <= [Agent] =>
  
  - Start : Start Position (0, 2)
  - Goal  : Goal Position (0, 6)
  - Puddle: Big Cost Position (0, 0)

- Agent:
  - Robot (Only left and right movement)

- Reward:
      |     0       1       2       3       4       5       6
  ----|------------------------------------------------------------
    0 | [  -100 ,  -1   ,   0   ,  -1   ,  -1   ,  -1   ,   10  ]

- Learning
  - max episodes: 100
  - episode end conditions: When the agent reaches the Goal
      """

    print(op_env)

    print("----------\n")
    
    print("Result:\n")
    print(" - best episode number : ", self.best_episode_)
    print(" - best reward         : ", self.best_reward_)
    print(" - best places passed  : ", self.best_x_)
    print(" - best policy         : ", self.best_a_)
    print(" - best Q-values       : ", self.best_q_)

    self.GUIImage()
    

if __name__ == "__main__":
  rl = Q_Learning()

  while True:

    rl.Step()

    if rl.check_goal_ == True:  # エージェントがゴールした場合
      rl.RestartEpisode()  # 初期化

      if rl.check_rl_end_ == True:  # エピソードが最大まで進んだ場合
        rl.VisualizationResult()  # 学習結果の可視化
        break

      rl.episode_now_ += 1
    
    else:
      rl.step_total_ += 1
