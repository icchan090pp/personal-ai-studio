# DeeL Pyess v0.1.3 — ノートPC互換モード

「GPU未接続です」とだけ表示されていた問題を修正し、PCのGPU・ライブラリ・実行環境を診断する画面を追加しました。

- NVIDIAはCUDA、WindowsのAMD/IntelなどはDirectMLを選択できます。非対応機種ではCPUを使用します。
- 軽量画像モデルTiny-SDを追加。256pxで生成し、指定解像度に拡大します。DirectML/CUDAの演算失敗時はCPUで再試行し、実際に使った環境を結果へ記録します。
- 「PC環境・AI機能」または起動ウィンドウの「このPCのAI機能をセットアップ」で必要なライブラリを導入します。requirements-gpu.txtとrequirements-directml.txtはアプリに同梱。ファイル作成・Python操作は不要です。
- DirectMLはPyTorch 2.4.1の独立環境、CUDA/CPUはPyTorch 2.7.0の環境を使用し、互いを上書きしません。
- CUDAはNVIDIA製GPU向けです。AMD GPUにCUDA GPUをソフトウェアだけで追加することはできません。

Windows 10/11・64bit用の `DeeL-Pyess-Setup-v0.1.3-Windows-x64.exe` を実行してください。
旧版を終了して同じ場所へ上書き更新します。投稿データ・完成モデルは `%LOCALAPPDATA%\DeeL Pyess\data` に保持します。
起動後にAI機能をセットアップし、「作品をつくる」で「Tiny-SD / 軽量画像」を選択してください。初回は追加ライブラリと約1.3GBのモデルを取得します。

軽量画像はRAM 8GB以上が目安。CPUは処理に数分以上かかる場合があります。
DirectMLはDirectX 12対応GPU向けで、全GPU・全演算・全ドライバを保証するものではありません。
画像LoRA学習・動画生成/学習・大型画像モデルには引き続き十分なVRAMを備えたNVIDIA CUDA GPUが必要です。音声読み上げ・声質特徴抽出はCPUで利用できます。
EXEは未署名。AMD/Intel実機GPUでの生成品質・性能は未検証です。
