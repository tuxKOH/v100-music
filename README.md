# v100-music

用两张 NVIDIA Tesla V100 的 FP32 矩阵乘法，把显卡的线圈啸叫/板级共振当成一种“乐器”。项目默认提供一个由用户 MIDI 转换出的 Bad Apple!! 单声道示例谱。

> 这是实验性硬件噪声项目，不是音频设备。请先读完 [DISCLAIMER.md](DISCLAIMER.md)。

## 重要声明

- 发声来自 GPU、VRM、电源和机箱的机械/电气共振，频率、音量和稳定性不可预测；同一型号的不同机器也可能完全不同。
- `GPU-Util=100%` 不等于一定有声音。功耗限制、时钟、驱动、固件、供电纹波、温度和散热方式都会改变结果。
- 所有音高映射都是经验数据，通常不单调，可能出现振荡、跳音、谐波或突然变调；必须在自己的机器上用手机/调音器手动标注。
- 长时间高功率运行可能损坏 GPU、显存、主板、VRM、电源、泵/风扇或数据，也可能造成听力伤害。作者不对任何硬件损坏、数据损失、人身伤害或财产损失负责。
- 水冷只降低核心温度，不代表 VRM、显存和主板供电得到充分散热；保留必要风流。出现过热、异味、冒烟、花屏、掉卡或系统不稳定时立即停止。
- 不要无人值守运行，不要为了更响而解除功耗/温度保护。本仓库按“现状”提供，不作任何适用性或安全保证。

完整版本见 [DISCLAIMER.md](DISCLAIMER.md)。

## 工作原理

`probe.py` 和 `player.py` 在两张 GPU 上交替提交相同的 FP32 GEMM。矩阵尺寸改变后，cuBLAS 可能选择不同的 tile/kernel，板级电流和共振频谱也会改变。项目只使用 FP32；没有音频输出、PWM 或扬声器驱动。

## 安装

需要 Linux、Python 3.10+、两张 CUDA 可见的 V100（或自行修改代码适配其他 GPU）。先按 NVIDIA/CUDA 和 PyTorch 官方说明安装匹配驱动的 CUDA 版 PyTorch，再安装 Python 依赖：

```bash
python3 -m pip install -r requirements.txt
```

V100 的 PyTorch/CUDA 版本取决于你的驱动，故不在 `requirements.txt` 中硬编码 torch 版本。

## 快速开始

查看设备并扫描矩阵尺寸：

```bash
python3 probe.py --sweep --duration 6 --rest 3
```

播放默认 Bad Apple!! 示例：

```bash
python3 player.py --score bad_apple.score.json
```

先只打印映射、不占用 GPU：

```bash
python3 player.py --score bad_apple.score.json --dry-run
```

默认使用 `cuda:0,cuda:1`。设备不同可显式指定：

```bash
python3 player.py --score bad_apple.score.json --devices 1,2
```

## 本机校准

不要直接相信仓库里的音高。用自己的手机调音器/录音，选择不同尺寸运行 `probe.py`，再用：

```bash
python3 gpu_tuner_gui.py
```

GUI 让手机播放目标音，GPU 播放参考音；点击“GPU 高了/低了”进行二分，并可用 `±8` 做手动微调。结果写入 `tuning.json`。

`tuning.json` 支持按音名或事件覆盖：

```json
{
  "notes": {"F#5": {"semitones": 1}, "B5": {"size_offset": -16}},
  "events": {"97": {"size": 1120}}
}
```

`size` 是直接指定矩阵边长；`size_offset` 以 8 为步长取整；`semitones` 是频率补偿。修改后请先 `--dry-run` 检查，再短时间实测。

## MIDI 转换

转换器不依赖 `mido`，会读取 tempo、Note On/Off 和 running status。MIDI 可能包含多轨、和弦或版权内容；请选择单声道轨道并人工检查：

```bash
python3 midi_to_score.py your_track.mid --track 1 --transpose 24
python3 player.py --score your_track.score.json --dry-run
```

`bad_apple.score.json` 是本项目的演示 score（由用户提供的 MIDI 转换而来，升调 24 个半音/两个八度），不是官方发行物，也不保证版权状态。仓库不默认重新分发原始 MIDI。

## 手机录音与降噪（可选）

```bash
python3 denoise.py 512.wav 768.wav 1024.wav 1536.wav 2048.wav
python3 pitch_gui.py
```

录音、降噪结果和机器专属的 `pitch_labels.json` 默认被 `.gitignore` 排除；请自行决定是否发布。

### 纯计算啸叫候选分析

录音无法证明声音来自某一颗电容；下面的工具只报告稳定、窄带的候选峰，不会修改音频：

```bash
python3 whine_pitch.py 512.wav 768.wav --top 12
```

它使用 STFT、中值谱、局部峰值和时间占用率。风扇的宽带风声通常得分较低，但风扇叶片的纯音也可能入选，因此仍需用手机调音器和不同负载录音人工确认。`denoise.py` 同样采用保守掩膜并保留窄带峰；没有能保证“只过滤风扇、零误杀啸叫”的通用模型。

想单独试听这些稳定窄带峰：

```bash
python3 whine_pitch.py 512.wav 768.wav --top 12 --extract-dir extracted
```

输出的 `extracted/*_stable_whine.wav` 只保留候选峰的窄带邻域，原始 WAV 不会被覆盖。它是试听和人工判断用的实验结果，不是安全的自动风扇/啸叫分类器。

如果能单独录一段只有风扇的 `fan.wav`，优先使用参考降噪：

```bash
python3 fan_denoise.py fan.wav 512.wav 768.wav 1024.wav 1536.wav 2048.wav
```

它按 `fan.wav` 的频谱做保守谱减，并给尖锐窄带峰保留最低增益；`--strength 0.5` 更温和，`1.0` 更激进。风扇的稳定叶片音仍可能保留，结果必须试听确认。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `probe.py` | FP32 双卡负载/尺寸扫描 |
| `player.py` | 根据频率或 score 播放音符 |
| `midi_to_score.py` | 标准 MIDI → JSON score |
| `gpu_tuner_gui.py` | 手机目标音反向标注矩阵尺寸 |
| `pitch_gui.py` | 录音与参考音比较标注 |
| `denoise.py` | 多段手机录音的频谱降噪 |
| `whine_pitch.py` | 纯计算提取稳定窄带音高候选 |
| `fan_denoise.py` | 使用独立风扇参考录音的保守降噪 |
| `bad_apple.score.json` | 默认演示谱 |
| `tuning.json` | 当前机器的示例校准，换机器必须重做 |

## 开发与贡献

提交前运行：

```bash
python3 -m py_compile probe.py player.py midi_to_score.py denoise.py pitch_gui.py gpu_tuner_gui.py
python3 player.py --score bad_apple.score.json --dry-run
```

欢迎提交新的校准方法、驱动/板卡对比和安全改进；不要提交原始录音、私有 MIDI 或包含个人信息的文件。

## 许可证

代码以 GNU GPL v3.0 许可证发布，见 [LICENSE](LICENSE)。示例谱、录音和其他第三方内容可能有独立版权；发布前请确认你拥有相应权利。GPL 仅覆盖本仓库中明确属于项目代码的部分，不会替你取得第三方素材的授权。
