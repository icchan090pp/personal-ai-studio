# DeeL Pyess v0.1.2 — Windows application

Windows向け配布をソースZIPから、Python同梱のセットアップEXEに修正しました。

- `DeeL-Pyess-Setup-v0.1.2-Windows-x64.exe` を実行してインストール。Pythonの別途インストール・ZIP展開は不要です。
- デスクトップとスタートメニューに「DeeL Pyess」のショートカットを作成します。
- 起動ウィンドウからブラウザが開きます。画像・動画、または日本語音声・声質機能はボタンで追加できます。
- データと診断ログは `%LOCALAPPDATA%\DeeL Pyess` に保存します。アンインストールでもデータを保持します。
- 起動ウィンドウを閉じるとサーバーも停止します。Windows 10 / 11、64bit向け、管理者権限不要です。
- Windows CIで実際のセットアップ、空白・日本語を含む保存先、同梱Pythonの整合性、EXEによるAPI起動・終了5回、アンインストールとデータ保持を検証します。
- ソースZIPはLinux・開発者用として引き続き提供します。配布サイトのWindowsボタンはEXEをダウンロードします。

GPU画像・動画学習/生成にはNVIDIA CUDA GPUと追加ライブラリ・モデル重みのダウンロードが必要です。
セットアップ本体に大型モデルは含まれません。SDXL、SSD-1B、PixArt Sigma、LTX-Videoは生成専用です。
gpt-ossは別途ローカルOllamaで動き、テキスト相談用です。
KokoroとOpenVoiceの実CPU動作を確認済み。OpenVoiceは参照声質特徴の取り込みで、全重みの追加学習ではありません。
AudioLDM2 LargeはCC-BY-NC-SA 4.0の非商用モデルです。4Kは拡大出力です。

初期検証版です。EXEは未署名です。CUDA実機の生成品質・実際の声の一致度・個人PCでのGPU追加機能は未検証です。
