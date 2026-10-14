# v100-music

用两张 NVIDIA Tesla V100 的计算负载，把显卡的线圈啸叫/板级共振当成一种“乐器”。现有方式通过 FP32 矩阵尺寸寻找音高，另有按目标 Hz 调制 GPU 活动的实验模式。项目默认提供一个由用户 MIDI 转换出的 Bad Apple!! 单声道示例谱。

> 这是实验性硬件噪声项目，不是音频设备。请先读完 [DISCLAIMER.md](DISCLAIMER.md)。

## 重要声明

- 发声来自 GPU、VRM、电源和机箱的机械/电气共振，频率、音量和稳定性不可预测；同一型号的不同机器也可能完全不同。
- `GPU-Util=100%` 不等于一定有声音。功耗限制、时钟、驱动、固件、供电纹波、温度和散热方式都会改变结果。
- 矩阵尺寸模式的音高映射是经验数据，通常不单调，可能出现振荡、跳音、谐波或突然变调；必须在自己的机器上用手机/调音器手动标注。按 Hz 调制负载的模式省去了尺寸标注，但仍需录音确认实际音高。
- 长时间高功率运行可能损坏 GPU、显存、主板、VRM、电源、泵/风扇或数据，也可能造成听力伤害。作者不对任何硬件损坏、数据损失、人身伤害或财产损失负责。
- 水冷只降低核心温度，不代表 VRM、显存和主板供电得到充分散热；保留必要风流。出现过热、异味、冒烟、花屏、掉卡或系统不稳定时立即停止。
- 不要无人值守运行，不要为了更响而解除功耗/温度保护。本仓库按“现状”提供，不作任何适用性或安全保证。

完整版本见 [DISCLAIMER.md](DISCLAIMER.md)。

## 工作原理

`probe.py` 和 `player.py` 在两张 GPU 上交替提交 FP32 GEMM。矩阵尺寸改变后，cuBLAS 可能选择不同的 tile/kernel，板级电流和共振频谱也会改变。

`power_tone.py` 则用 GPU 内部计时器，在目标频率下交替执行高、低活动量的代码。它尝试通过周期性负载引出对应的供电共振，不依赖矩阵尺寸—音高标注。CUDA 无法直接指定 VRM 电压、开关频率或电流波形；代码控制的只是工作负载。声音来自硬件振动，Line Out 仅用于可选的开场定位音。

## 安装

需要 Linux、Python 3.10+、两张 CUDA 可见的 V100（或自行修改代码适配其他 GPU）。先按 NVIDIA/CUDA 和 PyTorch 官方说明安装匹配驱动的 CUDA 版 PyTorch，再安装 Python 依赖：

```bash
python3 -m pip install -r requirements.txt
```

V100 的 PyTorch/CUDA 版本取决于你的驱动，故不在 `requirements.txt` 中硬编码 torch 版本。

视频降噪需要系统安装 FFmpeg（含 `ffprobe`）；开场定位音可选用系统的 `paplay` 或 `aplay`。

## 复现本机演奏

本机这次录制使用 `burst` 调度、FP16 Tensor Core 负载、`--work-repeats 512` 和 `--duty 0.5`。先安装下节说明的 CUDA Toolkit、C++ 编译器与 `ninja`，再用短音判断自己的机器是否响应：

```bash
python3 power_tone.py --engine burst --carrier tensor \
  --sweep 440,660,880 --duration 2 --rest 1 --devices 0,1 \
  --work-repeats 512 --duty 0.5
```

播放仓库自带的 Bad Apple!! 示例：

```bash
python3 power_tone.py --engine burst --carrier tensor \
  --score bad_apple.score.json --legato --devices 0,1 \
  --work-repeats 512 --duty 0.5 --countdown 10 --manifest bad_apple_take.json
```

这个示例让两张卡演奏同一旋律；千本樱的独立双声部和完整演奏命令见下方“卡 1 主旋律、卡 2 原谱伴奏”。本机的成功录音不代表其他机器也能得到相同音量或音准。

## 按目标频率调制负载（实验）

这个模式需要另外安装与 PyTorch 兼容的 CUDA Toolkit（含 `nvcc`）、C++ 编译器和 `ninja`。只有 CUDA 版 PyTorch 不够。首次运行会把 `native/power_tone.cu` 编译到 `.torch_extensions/`，后续复用缓存。支持一张或两张 compute capability >= 7.0 的 GPU；默认使用 `0,1`。

```bash
python3 -m pip install ninja
```

先用三段短音判断声音是否跟着调制频率走：

```bash
python3 power_tone.py --sweep 440,660,880 --duration 3 --rest 1 --manifest power_test.json
```

默认 `--carrier mixed` 混合计算与显存读写。也可以用相同扫频分别尝试 `--carrier compute` 和 `--carrier memory`。它们改变负载来源；某种负载功耗更高，不代表它一定更响。显存缓冲区在 GPU 上分配，播放期间没有 CPU—显存数据搬运。

### 模拟四位权重推理负载

`--carrier tensor` 在忙半周期执行真正的 FP16 Tensor Core 矩阵乘加；`--carrier fp4` 则循环读取合成的 E2M1 四位权重和 E4M3 分组缩放，解码成 FP16，再执行 Tensor Core 矩阵乘加。V100 没有原生 FP4 Tensor Core，这里使用 FP16 计算。该实验参考本机 NInfer 的负载组成，但没有加载模型，也没有复刻其完整层形状、调度、注意力或 TP2 allreduce。

先比较三种负载的同一个音：

```bash
python3 power_tone.py --carrier mixed --frequency 880 --duration 3
python3 power_tone.py --carrier tensor --frequency 880 --duration 3
python3 power_tone.py --carrier fp4 --frequency 880 --duration 3 --memory-mib 512
```

FP4 模式下，`--memory-mib` 是每张卡实际循环读取的权重工作集。它不是单纯的占位张量；占用显存不等于产生显存带宽。双 16 GiB 卡可进一步试接近模型容量的工作集，并添加通信负载：

```bash
python3 power_tone.py --carrier fp4 --sweep 440,660,880 --duration 3 --rest 1 \
  --memory-mib 14336 --link-gib-s 1 --manifest fp4_test.json
```

这里每张卡分配约 14 GiB；程序会按当前空闲显存检查，至少留下 256 MiB 余量，空间不足时请降低数值。它不会释放其他程序的显存。`--tensor-reuse 8` 是每次解码后复用 tile 的 Tensor Core 运算重复上限；降低到 `1` 会偏重读取和反量化，增加到 `32` 会偏重计算。哪个更响需要试听。

反量化、显存访问和通信都会增加切换延迟。本机在 10 kHz、`--tensor-reuse 32` 与通信并发时出现过漏周期；降低复用数或改用纯 `tensor` 模式可作对照。终端会提示采样切换率偏差超过 1% 的事件，不能把异常时序当成准确音高。

`--link-gib-s 1` 表示两方向合计目标 1 GiB/s。独立通信流反复复制不变的源缓冲区，播放前检查两个方向的数据；最终显示完成的拷贝字节数除以运行时间。它不强制打开 PCIe peer access，CUDA 可能通过主机暂存；这个数字不是 PCIe 仪表读数，也不是 NInfer 的真实 allreduce 测量。通信在休止和低活动阶段也运行，可能增加基线噪声、降低调制对比度，因此建议与 `--link-gib-s 0` 对照。

这两种 carrier 同样支持 score、双卡音程、移调和 Ctrl-C。

`--work-repeats 128` 控制每轮高负载阶段的运算重复上限（默认值）；FP4 模式下是连续处理的权重 tile 数上限。例如 `--carrier compute --work-repeats 256` 可以减少外层调度开销、增加计算强度；内核也会在运算组之间检查忙半周期的截止时间。计算强度和实际功耗的关系需要实测，可从 128、256 逐步试。

`--duty 0.5` 表示每个周期一半时间忙、一半时间减少活动；可以尝试 `0.25` 或 `0.75`。`--duty 1` 是持续负载对照，不产生这个忙闲调制。程序不修改功率上限、时钟或电压。默认驻留模式在低活动期间仍保留计算内核，不等于 GPU 完全空闲。

### 每个周期退出计算内核

如果 FP4 与纯 Tensor Core 听起来相同、声音都偏小，可以尝试改变调度方式：

```bash
python3 power_tone.py --engine burst --carrier tensor --sweep 440,660,880 \
  --duration 3 --rest 1 --work-repeats 512 --manifest burst_test.json
```

默认 `--engine resident` 在低活动阶段仍保留所有计算 blocks。`--engine burst` 每个忙半周期结束后退出全卡计算内核，闲半周期只留一个线程等待下次截止时间；CUDA Graph 连续提交这些任务，周期以 GPU 计时器为准。它让闲阶段释放更多 SM 资源，但不保证 GPU 进入休眠，也不保证啸叫更响。支持 `tensor`、`fp4` 两种负载，以及同样的乐谱、休止和双卡音程参数。

先保持默认 `--link-gib-s 0`，避免背景通信填满闲半周期。新模式的内核启动开销会压缩忙半周期，频率太高或占空比太低时可能漏周期；检查终端计时诊断，再判断录音。旧模式继续可用，声音效果需在自己的机器上比较。

两张卡分别调制两个频率：

```bash
python3 power_tone.py --frequency 440 --chord-semitones 7 --duration 3
```

这会让 GPU0 以 440 Hz 调制，GPU1 以约 659.26 Hz 调制。它是固定音程的双声部；是否听到两个对应的音仍需实测。

直接用示例谱生成调制时间表和旋律：

```bash
python3 power_tone.py --score bad_apple.score.json --legato --dry-run
python3 power_tone.py --score bad_apple.score.json --legato --chord-semitones -7
```

新模式的 `--transpose` 按半音移调，也会处理谱里显式写出的 `frequency_hz`；例如 `--transpose -12` 降一个八度。不读取旧的 `pitch_labels.json` 或 `tuning.json`。

### 卡 1 主旋律、卡 2 原谱伴奏

`--dual-score` 支持两个独立声部。谱中每个事件的 `frequencies_hz` 数组分别对应 `--devices` 的第一、第二张卡；`null` 是该卡休止。两条声部保留同一个 MIDI 的绝对时间，伴奏的换音不会重启主旋律的周期时钟。双卡谱不使用 `--legato` 或 `--chord-semitones`。

《千本樱》示例谱转换自 [aregan 下载页](https://www.aregan.net/p/senbonzakura.html) 的 MIDI，原曲为黒うさP / WhiteFlame，初音未来演唱。第一张卡使用 `MIKU` 主旋律（track 6），升两个八度；第二张卡使用 `NOTE` 伴奏（track 5），升一个八度。当前一张卡同时发一个音，伴奏和弦保留最高音声部，其他和弦音不同时演奏。另有选取 `BASS` 最低音的试听版本。仓库提供转换后的示例 JSON，原始 MIDI 需自行下载。

先播放约 25 秒的副歌：

```bash
python3 power_tone.py --engine burst --carrier tensor \
  --dual-score senbonzakura.chorus.dual.score.json --devices 0,1 \
  --work-repeats 512 --duty 0.5 --countdown 10 --manifest senbonzakura_take.json
```

`senbonzakura.full.dual.score.json` 是约 3 分 58 秒的完整编排，主旋律在约 31 秒进入。全曲命令：

```bash
python3 power_tone.py --engine burst --carrier tensor \
  --dual-score senbonzakura.full.dual.score.json --devices 0,1 \
  --work-repeats 512 --duty 0.5 --countdown 10 \
  --manifest senbonzakura_full_take.json
```

`--manifest` 是播放诊断日志，必须使用与输入乐谱不同的文件名；程序会拒绝覆盖输入乐谱。`senbonzakura.chorus_bass.dual.score.json` 则让第二张卡播放低音伴奏，移高三个八度。

从本地 MIDI 重新生成双卡谱及副歌截取：

```bash
python3 midi_to_score.py senbonzakura.mid --track 6 --transpose 24 \
  --accompaniment-track 5 --accompaniment-transpose 12 --accompaniment-strategy highest \
  --start 56.10384 --end 81.03888 --output senbonzakura.chorus.dual.score.json
```

删掉 `--start`/`--end` 即保留完整曲目。每条轨道各自选音，再在同一时间轴上截取，避免丢掉主旋律前的休止导致错拍。原曲和 MIDI 的权利属于各自权利人；本项目 GPL-3.0-only 代码许可不涵盖它们，本地下载不代表获准重新发布原 MIDI。

`--lineout-marker` 可在开场通过默认系统音频设备播放一次定位音。`--manifest` 保存调度模式、每段目标频率、相对时间、定位音/提交内核的主机时间及 GPU 计时诊断；系统音频延迟和内核提交延迟仍会影响录音对齐。两种调度模式均可用 Ctrl-C 中止。

终端显示的“GPU 活动切换 Hz”来自一个 GPU 线程的计时采样，并不是实际电流或录音音高。准确切换不保证准确发声：硬件可能放大谐波、出现固定共振，或几乎不响应。请用手机录音确认；如果声音没有跟着目标频率变化，就不能把这份时间表当成已校准音高。

## 快速开始

查看设备并扫描矩阵尺寸：

```bash
python3 probe.py --sweep --duration 6 --rest 3
```

录音用逐步定位扫描（每个 8 的倍数一段）：

```bash
python3 probe.py --fine-sweep --duration 1 --rest 0.5 --lineout-marker --manifest sweep.json
```

这会依次运行 `512, 520, 528, ... 2048`。会在开场从电脑 Line Out 播放一次约 160 ms 的 1760 Hz 定位音；之后不重复播放，避免标记音污染每个 batch。`sweep.json` 记录定位音和负载的实际时间，便于把手机录音切回对应矩阵尺寸。若系统没有 `paplay`/`aplay`，程序会提示但仍继续扫描。总时长约为 `193 × (1 + 0.5)` 秒，即约 4 分 49 秒。

如果要实验矩形 GEMM（例如 `1×64×64`，即 `(1×64) @ (64×64)`），可以单独运行：

```bash
python3 probe.py --shape 1x64x64 --duration 10 --rest 2
```

矩形形状不是方阵尺寸的等价替代：计算量、cuBLAS kernel 和功耗路径都会改变，声音可能更弱或完全不同。它目前只用于探索，播放器尚未把矩形形状自动映射成音符；请先实测并重新标注。

播放默认 Bad Apple!! 示例：

```bash
python3 player.py --score bad_apple.score.json
```

双卡和弦示例（GPU1 播主音，GPU2 播低五度）：

```bash
python3 player.py --score bad_apple.score.json --chord-semitones -7 --dry-run
python3 player.py --score bad_apple.score.json --chord-semitones -7
```

`--chord-semitones -7` 表示第二张卡比主音低 7 个半音；两张卡会分别使用自己的矩阵尺寸。先用 `--dry-run` 检查映射，再短时间实测。

如果想让旋律连续不断，把 score 中音符之间的短休止并入前一个音符：

```bash
python3 player.py --score bad_apple.score.json --chord-semitones -7 --legato --gap 0
```

这会改变原谱的断奏节奏，适合试听共振曲线；不影响 score 文件本身。

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

### 视频底噪参考与二次降噪

`reference_denoise.py` 整理了实录中用过的窄带音调掩膜：按独立底噪录音估计噪声频谱，保留明显高于底噪的窄带线。`--variable-noise` 按频段匹配音量，并统计水泵、水流声的波动。它使用 CPU，按带重叠上下文的片段处理长录音，不合成或替换音乐。

用与演奏相同的手机位置、朝向、录像软件和泵/风扇设置录 60–120 秒纯底噪；放稳手机，避免说话、敲键盘和移动。参考片段可排除头尾操作声，但无法保证自动识别所有非机器声音。

先把演奏和纯底噪 MP4 转为相同采样率、声道数的浮点 WAV：

```bash
ffmpeg -n -i performance.mp4 -map 0:a:0 -vn -ar 48000 -ac 2 -c:a pcm_f32le performance.wav
ffmpeg -n -i noise.mp4 -map 0:a:0 -vn -ar 48000 -ac 2 -c:a pcm_f32le noise.wav
python3 reference_denoise.py noise.wav performance.wav \
  --reference-start 2 --reference-end 58 --output performance_clean.wav
```

截取终点按自己的录音长度调整。若有两段底噪，想用“第一段过滤第二段，留下的残差再过滤已经处理过的演奏”，可以这样做：

```bash
python3 reference_denoise.py noise.wav water.wav --output water_residual.wav
python3 reference_denoise.py water_residual.wav performance_clean.wav \
  --variable-noise --output performance_clean_twice.wav
```

输入都是未经自动提音量的浮点 WAV。两级过滤可能削弱低信噪比的音调，也未必明显改善水声；这次水泵实录的二次过滤对开头底噪只额外降低约 1.7 dB。不能保证只去除噪声且完全保留啸叫，优先比较一级和二级结果。

合回原画面时，先查看原音轨相对视频的起点：

```bash
ffprobe -v error -select_streams a:0 -show_entries stream=start_time \
  -of default=noprint_wrappers=1:nokey=1 performance.mp4
```

下面示例假设原音轨起点为 0 秒；非零时把 `-itsoffset 0` 改成查到的偏移。视频流直接复制，保留原画质、时长和旋转信息：

```bash
ffmpeg -n -i performance.mp4 -itsoffset 0 -i performance_clean_twice.wav \
  -map 0:v:0 -map 1:a:0 -map_metadata 0 \
  -c:v copy -c:a aac -b:a 256k -movflags +faststart performance_denoised.mp4
```

脚本不会自动提音量。需要时给最后一步添加 `-af volume=10dB` 等固定增益，先检查处理后的峰值并保留余量，避免削波。原视频、底噪、输出 WAV/MP4 和本机播放日志默认被忽略，不上传仓库。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `probe.py` | FP32 双卡负载/尺寸扫描 |
| `player.py` | 根据频率或 score 播放音符 |
| `power_tone.py` | 按目标 Hz 调制 GPU 活动，单音/扫频/双卡双声部实验 |
| `native/power_tone.cu` | 驻留/周期退出 CUDA 内核、FP4 解码/FP16 Tensor Core、双卡拷贝与计时 |
| `midi_to_score.py` | 标准 MIDI → JSON score |
| `gpu_tuner_gui.py` | 手机目标音反向标注矩阵尺寸 |
| `pitch_gui.py` | 录音与参考音比较标注 |
| `denoise.py` | 多段手机录音的频谱降噪 |
| `whine_pitch.py` | 纯计算提取稳定窄带音高候选 |
| `fan_denoise.py` | 使用独立风扇参考录音的保守降噪 |
| `reference_denoise.py` | 实录窄带音调掩膜、可选水声统计与二次参考降噪 |
| `sweep_pitch.py` | 按时间清单扣除风扇谱并生成尺寸—音高候选表 |
| `bad_apple.score.json` | 默认演示谱 |
| `senbonzakura.*.dual.score.json` | 千本樱双声部全曲、副歌与低音伴奏示例 |
| `tuning.json` | 当前机器的示例校准，换机器必须重做 |

## 开发与贡献

提交前运行：

```bash
python3 -m compileall -q -x '/\.torch_extensions/' .
python3 player.py --score bad_apple.score.json --dry-run
python3 power_tone.py --sweep 440,660,880 --dry-run
python3 power_tone.py --engine burst --carrier tensor \
  --dual-score senbonzakura.full.dual.score.json --devices 0,1 --dry-run
python3 -m unittest discover -s tests -p test_dual_score.py -v
python3 -m unittest discover -s tests -p test_reference_denoise.py -v
python3 -m unittest discover -s tests -p test_power_tone.py -v
```

前三项播放命令使用 `--dry-run`，不启动 GPU 负载。双声部和降噪测试使用 CPU；最后一项需要 CUDA Toolkit，检查 E2M1/E4M3 解码的全部编码组合、双卡 FP4 调制与通信并发，以及两张卡独立休止。

欢迎提交新的校准方法、驱动/板卡对比和安全改进；不要提交原始录音、私有 MIDI 或包含个人信息的文件。

## 许可证

代码以 GNU GPL v3.0 许可证发布，见 [LICENSE](LICENSE)。示例谱、录音和其他第三方内容可能有独立版权；发布前请确认你拥有相应权利。GPL 仅覆盖本仓库中明确属于项目代码的部分，不会替你取得第三方素材的授权。
