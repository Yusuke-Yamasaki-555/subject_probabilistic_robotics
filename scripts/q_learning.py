#!/usr/bin/env python3

import matplotlib as plt
import numpy as np

class Q_Learning():
  def __init__(self):
    # 開始位置（＝初期位置）の座標（タプル）
    # ゴール位置の座標（タプル）

    # alfaの定義

    # 報酬の変数（行列）

    # Q値の変数（行列）
    # 状態の変数（＝現在座標の変数。タプル）？
    # 行動の変数（＝速度の変数（X方向に+-1 or Y方向の+-1）。タプル）

    # 更新後のQ値の変数（行列）

    self.check_goal = False
    self.check_rl_end = False

    self.episode_max = 1000
    self.episode_now = 1

  def ResetVariable(self, hoge, huge, hage):
    # 引数に渡された変数を全て初期化

    return 0

  def RestartEpisode(self):
    # ゴールにたどり着いた場合に実行
      # Q値の行列以外の変数を初期値にリセット

    rlt = self.ResetVariable(None, None, None)

    return 0

  def Step(self):
    # 毎Stepの処理

    return 0

  def VisualizationResult(self):
    # 学習結果を升目上のマップの画像として可視化
      # 水たまりと普通で色分け
      # 最適経路で色分け
      # 各マスに各状態におけるQ値の最大値を表示
    
    return 0



if __name__ == "__main__":
  rl = Q_Learning()

  while True:
    rl.Step()

    if rl.check_goal == True:
      rl.RestartEpisode()

      if rl.check_rl_end == True:
        rl.VisualizationResult()
        break

