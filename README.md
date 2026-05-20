# Gomoku-AI

一个面向 15x15 五子棋的 AI 训练与自动对弈项目。当前主线是 AlphaZero 风格的策略-价值网络：模型通过 MCTS 生成训练目标，使用自我对弈、强模型对抗、GomokuZeroAI 教师蒸馏等方式训练，并可以在本地 GUI 或浏览器棋盘上落子。

## 当前能力

- 15x15 五子棋规则、胜负判定、候选落点。
- ResNet 策略-价值网络训练。
- 可选 Transformer 混合主干训练。
- MCTS 自我对弈强化学习。
- 与 frozen opponent 对抗训练：
  - `GomokuZeroAI/result_15x15/checkpoints/iter_0150_15x15.pt`
  - `alphazero-gomoku/temp/best.pth.tar`，注意当前该文件是 9x9。
- GomokuZeroAI teacher-vs-teacher 蒸馏，让本项目 student 模型旁观学习。
- 训练 loss/样本数曲线可视化。
- 训练对局 GIF 和最终棋盘 PNG 可视化。
- Tkinter 本地人机对弈。
- 浏览器棋盘识别与鼠标自动落子。
- 旧版线性特征模型保留，可用于快速调试。

## 项目结构

```text
.
├── train.py                       # 神经网络强化学习训练入口
├── distill_gomokuzero.py          # GomokuZeroAI teacher 蒸馏入口
├── play.py                        # 本地 Tkinter 人机对弈
├── browser_vision_bot.py          # 浏览器棋盘识别和自动点击
├── requirements.txt
├── models/                        # 本项目模型输出目录
├── runs/                          # 训练曲线和可视化输出目录
├── GomokuZeroAI/                  # 外部 GomokuZeroAI 项目
├── alphazero-gomoku/              # 外部 alphazero-gomoku 项目
└── gomoku_ai/
    ├── game.py                    # 棋盘规则
    ├── neural.py                  # ResNet/Transformer/GomokuZeroAI 兼容模型
    ├── mcts.py                    # 通用 MCTS
    ├── neural_train.py            # 自我对弈、对抗训练、样本增强
    ├── training_visualizer.py     # CSV/PNG 曲线输出
    ├── game_visualizer.py         # 对局 GIF/PNG 输出
    ├── gui.py                     # Tkinter GUI
    ├── model.py                   # 旧版线性模型
    └── self_play.py               # 旧版线性自博弈训练
```

## 环境安装

建议 Python 3.10+。安装依赖：

```powershell
pip install -r requirements.txt
```

核心依赖：

```text
torch
numpy
Pillow
pyautogui
matplotlib
```

如果有 NVIDIA GPU，确认 PyTorch 能识别 CUDA：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

训练时不传 `--device cpu` 会自动优先使用 CUDA。

## 模型文件

常用模型路径：

```text
models/gomoku_resnet.pth.tar       # 默认 ResNet 模型
models/gomoku_transformer.pth.tar  # 推荐单独保存 Transformer 模型
models/gomoku_policy.json          # 旧版线性模型
```

默认规则：

```text
不加 --fresh：如果 --model 已存在，就继续训练已有模型
加 --fresh：忽略旧模型，从头开始
换 --model：训练或继续另一个模型文件
```

ResNet 和 Transformer 不建议共用同一个 `.pth.tar` 文件。

## 推荐训练路线

如果目标是尽快得到一个能下的 15x15 模型，推荐顺序是：

```text
1. 用 distill_gomokuzero.py 先做教师蒸馏，让模型模仿 GomokuZeroAI。
2. 再用 train.py --gomokuzero-opponent 做混合强化学习。
3. 最后用 play.py 或 browser_vision_bot.py 实战测试。
```

## 快速 Smoke Test

只验证流程能跑，不追求棋力：

```powershell
python train.py --size 5 --iterations 1 --self-play-games 1 --epochs 1 --mcts-sims 2 --channels 8 --blocks 1 --batch-size 8 --model models/smoke_resnet.pth.tar --fresh --device cpu
```

## ResNet 自我对弈训练

默认架构是 ResNet policy-value 网络：

```powershell
python train.py --size 15 --iterations 20 --self-play-games 16 --mcts-sims 80 --visualize --visual-name resnet_selfplay_15x15
```

更认真但更慢：

```powershell
python train.py --size 15 --iterations 100 --self-play-games 64 --mcts-sims 200 --channels 128 --blocks 8 --reward-weight 0.05 --visualize --visual-name resnet_15x15_long
```

## Transformer 训练

Transformer 是可选混合架构：卷积 stem + residual blocks + Transformer Encoder + policy/value heads。

第一次训练 Transformer：

```powershell
python train.py --architecture transformer --channels 128 --blocks 4 --model models/gomoku_transformer.pth.tar --size 15 --iterations 20 --self-play-games 16 --mcts-sims 80 --fresh --visualize --visual-name transformer_selfplay_15x15
```

继续训练同一个 Transformer：

```powershell
python train.py --architecture transformer --channels 128 --blocks 4 --model models/gomoku_transformer.pth.tar --size 15 --iterations 10 --self-play-games 16 --mcts-sims 80 --visualize --visual-name transformer_continue
```

Transformer 更慢，也更吃数据。早期调试建议优先用 ResNet。

## GomokuZeroAI 教师蒸馏

蒸馏脚本的对局双方是：

```text
GomokuZeroAI teacher A vs GomokuZeroAI teacher B
本项目 student 模型旁观学习
```

student 不参与落子，只学习 teacher 每一步 MCTS 搜索出的策略分布和最终胜负价值。

首次蒸馏 Transformer：

```powershell
python distill_gomokuzero.py --student models/gomoku_transformer.pth.tar --architecture transformer --channels 128 --blocks 4 --teacher-a GomokuZeroAI/result_15x15/checkpoints/iter_0150_15x15.pt --games 20 --iterations 10 --mcts-sims 80 --visualize --visualize-games --visual-games-max 1 --fresh
```

继续蒸馏已有 Transformer，去掉 `--fresh`：

```powershell
python distill_gomokuzero.py --student models/gomoku_transformer.pth.tar --teacher-a GomokuZeroAI/result_15x15/checkpoints/iter_0150_15x15.pt --games 20 --iterations 10 --mcts-sims 80 --visualize --visual-name transformer_distill_continue
```

如果有两个不同 teacher：

```powershell
python distill_gomokuzero.py --teacher-a path/to/teacher_a.pt --teacher-b path/to/teacher_b.pt --student models/gomoku_transformer.pth.tar
```

### 只学习稳定步数

默认情况下，蒸馏只学习 `--temp-threshold` 之后的步数。比如：

```powershell
--temp-threshold 8
```

表示前 8 手 teacher 可以采样探索，但这些步数不进入 student 训练样本；第 9 手开始才学习。

手动指定过滤步数：

```powershell
python distill_gomokuzero.py --student models/gomoku_transformer.pth.tar --teacher-a GomokuZeroAI/result_15x15/checkpoints/iter_0150_15x15.pt --games 20 --iterations 10 --mcts-sims 80 --temp-threshold 8 --learn-after-step 8
```

如果想从第 1 手开始全部学习：

```powershell
--learn-after-step 0
```

### 蒸馏后观察 student

每轮蒸馏完成后，可以让 student 自己和自己下几局，只观察，不学习：

```powershell
python distill_gomokuzero.py --student models/gomoku_transformer.pth.tar --teacher-a GomokuZeroAI/result_15x15/checkpoints/iter_0150_15x15.pt --games 20 --iterations 10 --mcts-sims 80 --student-eval-games 2 --student-eval-mcts-sims 40 --visualize --visualize-games --visual-games-max 2 --visual-name transformer_distill_continue
```

输出示例：

```text
runs/distill_games/iter_0001/student_self_eval_game_001.gif
runs/distill_games/iter_0001/student_self_eval_game_001_final.png
```

## 与 GomokuZeroAI 对抗混合训练

`GomokuZeroAI/result_15x15/checkpoints/iter_0150_15x15.pt` 是 15x15 checkpoint，可以直接作为 frozen opponent。

ResNet 混合训练：

```powershell
python train.py --size 15 --gomokuzero-opponent GomokuZeroAI/result_15x15/checkpoints/iter_0150_15x15.pt --self-play-games 8 --opponent-games 8 --iterations 20 --mcts-sims 80 --opponent-mcts-sims 80 --reward-weight 0.05 --visualize --visual-name gz_mix_15x15
```

Transformer 混合训练：

```powershell
python train.py --architecture transformer --channels 128 --blocks 4 --model models/gomoku_transformer.pth.tar --size 15 --gomokuzero-opponent GomokuZeroAI/result_15x15/checkpoints/iter_0150_15x15.pt --self-play-games 5 --opponent-games 5 --iterations 10 --mcts-sims 80 --opponent-mcts-sims 80 --reward-weight 0.05 --visualize --visualize-games --visual-games-max 2 --visual-name gz_mix_15x15_transformer
```

`--temp-threshold` 会同时影响 self-play 和 opponent-play。前几手可以采样探索，之后使用贪心 MCTS。与 GomokuZeroAI/alphazero-gomoku 等外部模型对抗时，learner 和 opponent 双方使用相同的 `temp-threshold`。比如默认：

```powershell
--temp-threshold 4
```

表示第 1~4 手采样，第 5 手开始贪心。样本过滤规则默认也会跳过采样步：

```text
self-play：默认只学习 temp-threshold 之后的步数
opponent-play：默认只学习 temp-threshold 之后的步数
```

也可以手动指定统一过滤步数：

```powershell
python train.py --size 15 --gomokuzero-opponent GomokuZeroAI/result_15x15/checkpoints/iter_0150_15x15.pt --self-play-games 5 --opponent-games 5 --iterations 10 --mcts-sims 80 --temp-threshold 4 --learn-after-step 4
```

如果想让 self-play 和 opponent-play 都从第一手开始全部学习：

```powershell
--learn-after-step 0
```

对抗训练默认还会在 learner 输给外部对手时，额外学习对手赢局中的落子样本：

```text
learner 回合：始终保存，用来学自己为什么赢/输
opponent 回合：只有 opponent 最终赢了才保存，用来学对面为什么赢
```

这个行为默认开启。如果只想复盘 learner 自己的回合，不学习对手赢法，可以关闭：

```powershell
--no-learn-opponent-wins
```

对抗日志会显示 student 是否赢：

```text
iter 10 vs-GomokuZeroAI 5/5 | learner win | steps 83
iter 10 vs-GomokuZeroAI 5/5 | learner loss | steps 41
iter 10 vs-GomokuZeroAI 5/5 | draw | steps 225
```

## 与 alphazero-gomoku 对抗训练

当前 `alphazero-gomoku/temp/best.pth.tar` 检测为 9x9 模型，不能直接和 15x15 模型对抗。可以让当前模型尺寸匹配它：

```powershell
python train.py --match-opponent-size --alphazero-opponent alphazero-gomoku/temp/best.pth.tar --opponent-games 4 --iterations 10 --self-play-games 8 --mcts-sims 80 --opponent-mcts-sims 80 --fresh
```

如果以后换成 15x15 的 alphazero-gomoku checkpoint，就可以：

```powershell
python train.py --size 15 --alphazero-opponent alphazero-gomoku/temp/best.pth.tar --opponent-games 8
```

## 训练可视化

开启曲线输出：

```powershell
--visualize --visual-name my_run
```

输出：

```text
runs/my_run.csv
runs/my_run.png
runs/my_run_counts.png
```

说明：

```text
my_run.png         loss、reward_weight 等较小量级指标
my_run_counts.png examples、new_examples、opponent_examples、mcts_sims 等大数量级指标
```

开启对局 GIF：

```powershell
--visualize-games --visual-games-max 2
```

输出示例：

```text
runs/games/iter_0001/self_play_game_001.gif
runs/games/iter_0001/self_play_game_001_final.png
runs/games/iter_0001/vs_GomokuZeroAI_game_001.gif
runs/games/iter_0001/vs_GomokuZeroAI_game_001_final.png
```

控制生成频率：

```text
--visual-games-every 5   每 5 轮保存一次
--visual-games-max 1     每个保存轮最多保存 1 局
```

长训练时不要把 `--visual-games-max` 设太大，GIF 会拖慢训练并占磁盘。

## 常用训练参数

```text
--size                  棋盘尺寸，浏览器和本地当前主要用 15
--iterations            训练迭代轮数
--self-play-games       每轮自我对弈局数
--opponent-games        每轮与 frozen opponent 对抗局数
--mcts-sims             learner 每步 MCTS 模拟次数
--opponent-mcts-sims    opponent 每步 MCTS 模拟次数
--architecture          resnet 或 transformer
--channels              网络宽度
--blocks                residual blocks 数量
--lr                    学习率，默认 0.001
--reward-weight         棋形奖励塑形权重
--learn-after-step      只学习该步数之后的训练样本，默认等于 temp-threshold
--learn-opponent-wins   对抗失败时额外学习对手赢局中的落子，默认开启
--fresh                 从头训练
--model                 模型保存/加载路径
--device                cuda 或 cpu
```

`mcts-sims` 越大，样本质量越高，但速度越慢。快速实验建议 `20~40`，中等实验 `80`，认真训练再考虑 `200+`。

`reward-weight` 建议从小到大试：

```text
0       不使用棋形奖励
0.03    很轻的棋形引导
0.05    推荐起步
0.08    更明显
0.1~0.2 较激进，可能过度追逐局部棋形
```

## 模型学到的样本

神经网络训练样本是：

```python
(board, pi, value)
```

含义：

```text
board：当前玩家视角的棋盘
pi：MCTS 搜索后的落子概率分布
value：最终胜负价值，从当前玩家视角计算
```

15x15 棋盘上，`pi` 长度是 225：

```text
move_index = row * 15 + col
```

当前项目的 student 网络输入是 3 通道：

```text
channel 0：当前玩家棋子
channel 1：对手棋子
channel 2：空位
```

每个样本会做 8 倍对称增强：

```text
旋转 0/90/180/270 度
每个旋转再左右翻转
```

棋盘和 `pi` 会同步变换，落点概率不会错位。

## 人机对弈

启动本地 Tkinter GUI：

```powershell
python play.py
```

默认加载顺序：

```text
models/gomoku_resnet.pth.tar
models/gomoku_policy.json
内置线性权重
```

如果要加载 Transformer，可以在 GUI 里点“加载模型”，选择：

```text
models/gomoku_transformer.pth.tar
```

## 浏览器棋盘识别和自动落子

脚本会截图识别网页棋盘，红色棋子作为对方，绿色棋子作为本方 AI；检测到红色棋子数量增加后，计算下一手并鼠标左键点击。

你当前校准过的 15x15 浏览器棋盘参数：

```powershell
python browser_vision_bot.py --board-size 15 --board-left 1052 --board-top 535 --cell 35
```

加载 Transformer 模型：

```powershell
python browser_vision_bot.py --model models/gomoku_transformer.pth.tar --board-size 15 --board-left 1052 --board-top 535 --cell 35
```

先干跑，不点击：

```powershell
python browser_vision_bot.py --model models/gomoku_transformer.pth.tar --board-size 15 --board-left 1052 --board-top 535 --cell 35 --once --dry-run --debug-image debug/board_detected.png
```

如果浏览器窗口位置、缩放比例、页面滚动位置改变，`--board-left`、`--board-top`、`--cell` 需要重新校准。

安全停止：

```text
把鼠标移动到屏幕左上角，触发 pyautogui failsafe
或在终端按 Ctrl+C
```

请只在你有权限控制的页面或自己的测试对局里使用自动点击。

## 旧版线性模型

旧版线性模型用于快速调试，不依赖神经网络：

```powershell
python train.py --legacy-linear --episodes 2000 --model models/gomoku_policy.json --fresh --visualize --visual-name linear_15x15
```

它使用手工特征：

```text
中心偏好
邻近棋子
己方/对方最大连线
活二、活三、活四、冲四
直接成五、立即阻挡成五
```

## 常见问题

### 重复训练会接着上一次吗

会。只要模型文件存在，且不加 `--fresh`，就会继续训练。

```powershell
python train.py --model models/gomoku_transformer.pth.tar
```

从头训练：

```powershell
python train.py --model models/gomoku_transformer.pth.tar --fresh
```

### Transformer warning 是错误吗

如果看到：

```text
enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
```

这是 PyTorch Transformer 的性能提示，不影响运行。

### 训练太慢怎么办

先降低搜索和可视化：

```powershell
--mcts-sims 20 --opponent-mcts-sims 20
```

关闭或减少 GIF：

```powershell
不加 --visualize-games
或 --visual-games-max 1
```

先用小模型：

```powershell
--channels 64 --blocks 4
```

先用 ResNet，不要一开始就用 Transformer。

### 为什么 teacher 每局完全一样

如果蒸馏时使用：

```powershell
--temp-threshold 0
```

teacher 从第一手开始完全贪心，且 teacher-a 和 teacher-b 是同一个 checkpoint，就会产生确定性对局。要增加多样性，用：

```powershell
--temp-threshold 2
```

或：

```powershell
--temp-threshold 4
```

### 为什么开启 temp 后一方看起来很笨

前几手采样会引入随机性，五子棋开局又很敏感，一方采样到差开局就会被滚雪球。折中方案：

```powershell
--temp-threshold 2 --mcts-sims 200
```

或者只学习采样步数之后的样本：

```powershell
--temp-threshold 4 --learn-after-step 4
```

## 算法简述

主线训练是 AlphaZero 风格：

```text
当前模型评估局面
MCTS 根据模型 policy/value 搜索更好的落子分布
用搜索后的 pi 选择落子
对局结束后得到 value
训练模型拟合 pi 和 value
新模型继续生成更好的数据
```

训练目标：

```text
policy_loss = 模型落子概率 vs MCTS 落子分布
value_loss = 模型局面价值 vs 最终胜负
```

蒸馏训练则是：

```text
GomokuZeroAI teacher 负责下棋和生成 pi
本项目 student 只学习 teacher 样本
student 不参与 teacher 对局
```

蒸馏适合预热，对抗和自我对弈适合后续强化。

## 致谢

感谢以下项目提供的参考与启发：

- [Nagi-ovo/alphazero-gomoku](https://github.com/Nagi-ovo/alphazero-gomoku)
- [maojh15/GomokuZeroAI](https://github.com/maojh15/GomokuZeroAI)
