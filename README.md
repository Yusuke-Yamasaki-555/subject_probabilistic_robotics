### LICENSE & Reference

#### LICENSE

This repository is licensed under the MIT License.<br>
See the `LICENSE` file for more details.

Copyright (C) 2024 Yusuke Yamasaki. All Rights Reserved.

#### References

To understand the theory and implementation, I referred to the following books and websites.
- 斉藤康毅, “ゼロから作るDeep Learning 4 - 強化学習編”, オライリー・ジャパン, 2022
- 上田隆一, “詳解　確率ロボティクス　Python による基礎あるごりずむの実装”, 講談社, 2020
- ryuichiueda, "slides_marp/prob_robotics_2024", GitHub, https://github.com/ryuichiueda/slides_marp/tree/master/prob_robotics_2024, last viewed on: 2025/1/10
  - And, the content of the lecture using these slides.

In addition, the program to display the images was implemented with reference to the following site.<br>
See the `scripts/q_learning.py` file for more details.
 - matplotlib, "Annotated heatmap", https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py, last viewed on: 2025/1/10


---

<br><br>

# Q学習の実装

## 概要



## 処理内容



## 実行方法

以下のコマンドで実行することができます（Linux系OSの場合）。

```bash
python3 script/q_learning.py
```

## 実行結果

実行することで、学習終了後に以下のような標準出力が得られます（学習結果は実行毎に異なります）。

```bash

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
      
----------

Result:

 - best episode number :  5
 - best reward         :  7
 - best places passed  :  [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6]]
 - best policy         :  [[0, 1], [0, 1], [0, 1], [0, 1], [0, -1]]
 - best Q-values       :  [[-67, 0, 5, 6, 10, 10, 17]]

```

これは、学習時の環境やエージェントの設定に加えて学習結果を確認することができます。

また、同時に以下のような図を含むウィンドウが表示されます。

![q values](logs/output.png)

これは、環境である１x６個の各マスに対応した、一番初めに最良であったエピソードの際に得られた Q 値を表示したものになっています。各マスの色は Q 値に対応しています。


以上２つの出力から、スタート位置である [0,2] からゴール位置である [0,6] に向かうにつれて値が大きくなり、水たまりの位置である [0,0] では値が大きく減少しています。このことから、スタート位置から右方向に進み続けてゴール位置にたどり着く方策が最適な方策として得られたことがわかります。