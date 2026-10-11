# DeeL Pyess

独自の画像・動画モデルを育てるローカルWebアプリ。PCと、同じWi-Fiのスマートフォンから操作できます。
初期検証版です。**100件の製品試験を5回連続で実行**し、全件合格しました。検証スクリプトとJUnit記録を同梱します。
CUDA実機・実サイズの基盤モデルによる学習／生成品質・個人のWindows実機でのGPU機能は未検証です。Windowsセットアップと基本起動はWindows CIで確認しました。

## 機能

- SD 1.5画像LoRA、CogVideoX-5b-I2V動画LoRAと生成。実験用の短い効果音・環境音モデルの学習／動画への付加
- エクスプローラー・フォルダ投稿、グルーピング、指定サイト内収集、Wikimedia Commons検索、出典保存
- 画像は通常・半教師ありBLIP・スタイル・RWR・生成テスト・写し学習。動画／音声は説明付き通常・写し・スタイル学習
- 学習チャットと生成チャットを分離し、モデルごとの履歴を保存
- 初期ユーザールールは空。歯車から共通生成プロンプトを設定・リセット
- 画像／動画ともノーマル20、パワー35、エクセレント（深考）50、ウルトラ（レベル200）200ステップの生成プリセット
- 指定サイズ2100×2050・720p〜4KのPNG／MP4書き出し。高解像度は生成後のLanczos拡大と余白調整
- 実測の損失・勾配・学習重みL2を棒グラフ表示。8つのAI開発実践コースと実験ノート

[操作ガイド](docs/usage.md) / [試験結果・検証範囲](docs/verification.md)

## Windowsへのインストール・更新

[Releases](https://github.com/icchan090pp/personal-ai-studio/releases) から `DeeL-Pyess-Setup-v0.1.3-Windows-x64.exe` を取得して実行してください。
Windows 10/11・64bit用、Python同梱です。既存のDeeL Pyessを終了し、同じ保存先へ上書き更新できます。
デスクトップの「DeeL Pyess」で起動し、「このPCのAI機能をセットアップ」を実行します。
投稿データ・完成モデルは `%LOCALAPPDATA%\DeeL Pyess\data`、診断ログは同フォルダのstartup.logです。
アンインストールでもデータを保持します。EXEは未署名です。

### 一般のノートPC向け（v0.1.3）

NVIDIA CUDA、Windows DirectML（AMD/Intelなど）、CPUを区別して診断します。
AMD/IntelはCUDAを使用できません。DirectMLは独立環境に導入し、CUDA/CPU用のPyTorchを上書きしません。
「作品をつくる」で「Tiny-SD / 軽量画像」を選ぶと、256pxの画像を生成して指定サイズへ拡大します。
GPUの演算非対応・メモリ不足時はCPUで再試行し、使用した環境を結果に記録します。
初回セットアップは数GBのダウンロード、Tiny-SDの重みは約1.3GBです。RAM 8GB以上を目安としてください。
全GPU・全演算の保証はできません。画像LoRA学習、動画、大型モデルは十分なVRAMを持つCUDA GPUが必要です。
日本語音声と声質特徴抽出はCPUでも利用できます。CPUでは処理に時間がかかります。

### Linux・開発者向けソース版

ソースZIPを展開し、Python 3.12/3.13で `./install.sh`、`./start.sh` を実行します。
AI機能は仮想環境で `python -m studio.setup_runtime --backend cpu`、NVIDIAは `--backend cuda` を指定できます。
日本語音声・声質の追加は `--voice`。Windowsのソース版はInstall-Windows.bat/Start-Windows.batを利用できます。
スマートフォン用はソース版の `Start-Phone.bat`／`./start-phone.sh` を実行し、同じ信頼できるWi-Fiの接続URLを開きます。

## モデルと環境

画像モデル：`stable-diffusion-v1-5/stable-diffusion-v1-5`。
動画モデル：`THUDM/CogVideoX-5b-I2V`。モデル重みはZIPに含まず、初回に配布元から取得します。
それぞれのモデルライセンス・アクセス条件が適用されます。
目安：画像VRAM 8GB以上、動画生成16〜24GB以上、動画LoRA学習48〜80GB、RAM32GB以上。
実際の必要量は機器・モデル・設定で変わります。動画学習では特に大きいGPUとホストRAMが必要です。

[Stability Matrix Releases](https://github.com/LykosAI/StabilityMatrix/releases) はモデル管理アプリの配布元です。
DeeL Pyessは独立アプリとして、管理フォルダにあるSD 1.5 `.safetensors` を読み込みます。
Windows PowerShellでは `$env:STUDIO_CHECKPOINT="C:\Models\SD15.safetensors"`、
Linuxでは `STUDIO_CHECKPOINT=/path/to/SD15.safetensors ./start.sh`。SDXL／Flux／SD2は対象外です。
画像LoRAはDiffusers形式とKohya形式、動画LoRAはCogVideoX Diffusers形式を出力します。

実験用の音声学習はゼロから学ぶ小型の波形拡散モデルです。短い効果音の実験用で、自然な会話・声の再現・高品質音楽は対象外。
学習と生成に同じ説明文を使う条件ID方式で、文章の意味理解や映像との音声同期はしません。
RWRは報酬重み付き回帰で、PPO／DDPOではありません。深考は生成ステップの多いプリセットです。
ユーザールールのリセットは共通プロンプトを空にする機能で、基盤モデルの性質は消せません。

## 開発と検証

```bash
bash scripts/cloud-environment.sh setup
bash scripts/cloud-environment.sh verify
```

5回の各実行で、ちょうど100件・失敗0・エラー0・スキップ0をJUnitで検証します。
CPU上の実際の小型SD／CogVideoX／音声ネットワークで、最適化・保存・読み込み・生成を試験。
PNG／4K MP4／AAC音声付加、HTTP API、PC／モバイルの実ブラウザ操作と配布サイトを確認します。
実基盤モデルのGPU動作と品質は別の実機検証が必要です。失敗を隠す代替生成・架空の学習成功はありません。

ライセンス：アプリはMIT。データ・モデルにはそれぞれの権利とライセンスが適用されます。

Linuxで実行権限がない場合：`chmod +x install.sh start.sh start-phone.sh` を実行してください。

v0.1.2では、生成専用の公開モデル4種類（SDXL、SSD-1B、PixArt Sigma、LTX-Video）、
ローカルOllama経由のOpenAI gpt-oss-20b/120b学習相談、ウルトラ（画像・動画200ステップ）、
データを保持する学習停止・リセットを追加しました。追加4モデルのLoRA学習は未対応です。
公開モデルの実サイズGPU生成・gpt-ossの実推論は本CPU開発環境では未検証です。

音声生成にAudioLDM2 Large（非商用の効果音）とKokoro 82M（日本語読み上げ）を追加。
OpenVoice V2で参照音声の声質特徴を取り込み、速度・高さの対応指示を保存して読み上げへ反映。
動画投稿とサイト収集から音声を抽出し、音声・声質・動画学習へ利用できます。
Windows版の音声付きセットアップ、またはLinux `./install.sh --gpu --voice` で追加音声ライブラリを導入します。
声質の取り込みは特徴抽出で、モデル全体の追加学習ではありません。再現度はモデルと録音によって変わります。
