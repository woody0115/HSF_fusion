# HSF_fusion — Event + RGB 融合物件偵測（自駕車）

重現並延伸 *Hierarchical Spatiotemporal Fusion for Event-Visible Object Detection*
（Jhong et al., ICRA 2025）。論文沒有公開程式碼，這裡依論文描述自行實作。

- 資料集：DSEC-Detection（8 類：pedestrian, rider, car, bus, truck, bicycle, motorcycle, train）
- 基礎模型：雙流 YOLOv8s（Ultralytics 8.3.158 的元件、loss、評估函式；不修改 Ultralytics 原始碼）
- 論文目標：mAP50 **67.2** / mAP50-95 **45.6**

## 進度

| 步驟 | 內容 | 狀態 |
|---|---|---|
| 0 | 下載 DSEC-Det 到 `D:\DSEC`（`tools/download_dsec_det.ps1`，4 連線續傳；時間戳記另由 `tools/fetch_timestamps.py` 抽出） | ✅ |
| 1 | 環境與專案架構（`tools/check_env.py`） | ✅ |
| 2 | 資料前處理：GEAR event 影像、YOLO 標註（`tools/preprocess_dsec.py`；訓練資料在 E 槽 SSD） | ✅ |
| 3 | 單流基準：只用 RGB / 只用 event（論文 65.6/44.4 、 41.9/24.8） | ✅ RGB 75.6 / 53.9、event 51.4 / 32.9（自家評估器；RGB 訓練結束時 Ultralytics 驗證為 75.5 / 54.0） |
| 4 | 雙流基準：特徵相加（論文 64.6/43.3） | ✅ #3 e20 51.9；#6 lr 1e-3 53.0 |
| 5 | + AF 融合模組（論文 65.7/44.5） | ✅ #4 e20 52.9；#7 lr 1e-3 53.5 |
| 6 | + TFS 時間模組 → 完整 HSF（論文 67.2/45.6） | ✅ #5 e20 51.5；#8 lr 1e-3 53.1 |
| 7 | 開放世界 DSEC-OWOD：Task 1–4（`configs/owod_tasks.yaml`），UniOW 式分類頭（CLIP 文字嵌入 + 萬用嵌入），指標 U-Recall / WI / A-OSE / PK·CK·Both mAP | ✅ #11（T3 U-Recall 6.7、CK 0.2） |
| 8 | 光線自適應融合（我們的延伸）：`iaf_add` / `iaf_af`，加上 hard switch 參考值（`route_*`） | ✅ #9、#10 都是 53.5；閘門沒學會看光線 |
| 9 | event 參與 unknown 判斷：做法 1 推論時依框內 event 密度重新評分（`tools/owod_event_rescore.py`）；做法 3 T1 訓練時 event 活躍的背景不當負樣本（`train_owod.py --event-ignore 0.1`） | ✅ #12（T3 U-Recall 26.0） |
| 10 | CK 崩潰診斷（線性探測、調嵌入訓練量、框召回、嵌入空間；`tools/diag_ck.py`、`tools/diag_data_stats.py`） | 進行中，結果在 `runs/diag/` |

## 螢幕影片 demo

```bat
.venv\Scripts\python.exe tools\demo_screen.py --source screen --model af
```

- **畫面來源**：啟動後用滑鼠框出影片播放的區域；`--source` 也可以給影片檔。
- **模型**：`--model af | add | iaf_add | rgb | owod`。
- **按鍵與滑桿**：q 結束、s 存圖、e 顯示／隱藏模擬 event、r 清空 event 記憶；滑桿是 conf、IoU、unknown。
- **沒有 event camera**：event 從畫面模擬（`hsf/data/event_sim.py`）。在 DSEC 上，模擬 event 的 AF 模型是 52.9，真實 event 53.5，只用 RGB 53.9（`tools/event_sim_check.py`），所以這個 demo 只用來展示運作方式。

## 已確認的設計決定
- **GEAR**：事件時窗 = 前一張影像 → 這一張影像（sync=front）；c=8、τ=10；tanh 前以灰階單位（÷255）縮放；
  h、feat 以中間灰為 0，num 以黑色為 0；h 照論文不額外衰減。
- **標註**：框裁到 640×480 內，裁後寬或高 < 2 px 丟棄；保留 track_id。class 7（train）在資料中沒有框，實際評估 7 類。
- **驗證**：用 13 個 test 序列（同論文的 47/13 切分）。
- **DSEC-OWOD Task**：T1 car → T2 bus、truck → T3 pedestrian、rider → T4 bicycle、motorcycle。
- 右相機 event 保留（未來可做立體）。
- **訓練資料放 E 槽（NVMe SSD）**：D 槽是傳統硬碟，大量小檔隨機讀取太慢；原始下載仍在 D 槽。
- **雙流訓練排程**：論文是 40 epochs；步驟 4–6 統一用 20 epochs（骨幹已由單流預訓練），
  其餘照論文（SGD lr 0.005、momentum 0.937、batch 8；Ultralytics 式 nominal batch 64 累積梯度、
  warmup、EMA、AMP）。實測約 84 ms/iter，約 12 分鐘/epoch（訓練時避免同時跑解壓或前處理）。
- **評估**：與 Ultralytics validator 相同（conf 0.001、IoU 0.7、multi-label、max 300）；
  `tools/verify_evaluator.py` 實測兩者 mAP 差 < 0.3 點。
- **AF 模組**（論文 III-D）：自適應混合池化（λa·avg + λm·max）→ 兩模態 token 串接 + 位置編碼 →
  1 層 transformer（λsa1/λsa2、λmlp1/λmlp2 自適應殘差）→ 上採樣加回兩個分支 → 串接 + 1×1 Conv 進 Neck。
  論文未寫、我們的選擇：池化到 8×8（同 CFT）、1 層、8 頭、MLP ×4；並在「加回分支」與「進 Neck」三處加上
  **初始為 0 的可學習權重**，使 AF 起始時等同相加基準（否則 Neck 一開始收到隨機特徵，mAP 從 0 重學）。
  參數 16.2 M → 21.2 M。
- **TFS 模組**（論文 III-C，式 6–9）：event 分支上的卷積 GRU，更新/重置閘門同時看 event 與 RGB 特徵，
  1×1 卷積；接在每個尺度的 AF 之前。我們的補充：輸出 x_evt + λ_t·h_t（λ_t 初始 0，理由同 AF）。
  **訓練**：4 張連續影格為一個片段（同一組資料增強），batch = 4 片段 × 4 張，片段內完整 BPTT、每片段記憶歸零；
  無標註的影格照常更新記憶但不算 loss。**評估**：每個 test 序列依時間順序串流、記憶不中斷（序列開頭歸零），
  只在有標註的影格算 mAP，每 2 個 epoch 評估一次。片段長度 4 與 batch 為我們的選擇（論文未寫）。
- **步驟 7（DSEC-OWOD）採高效 UniOW 流程**（`hsf/owod/`、`tools/train_owod.py`）：
  分類頭換成對比式（每個 anchor 的 512 維嵌入 · CLIP 文字嵌入，文字編碼器沿用 YOLO-UniOW 匯出的那份）；
  T1 只用 car 標註完整訓練（骨幹從 COCO 預訓練開始，**不用**看過全部類別的步驟 3–6 權重，避免洩漏），
  再調「object」萬用嵌入；每個 Task 模型凍結，只學新類別嵌入 + 「unknown」嵌入（目標 = object 分數，
  排除與當前標註重疊或舊類別高分的 anchor）；舊類別嵌入凍結 → 不遺忘、不需 exemplar replay。
  評估：mAP 用 AP50（OWOD 慣例），U-Recall（IoU 0.5）、WI（已知 recall 0.8）、A-OSE，每張最多 100 框，
  unknown 去重同 UniOW（IoU ≥ 0.99 且已知分數 > 0.2）。融合方式用 add（凍結調嵌入的階段不支援時間狀態）。
  已知弱點：凍結的特徵只在 car 上訓練過，後續新類別的 CK mAP 會受限（UniOW 靠大型預訓練彌補）。
- **修正紀錄（10/05）**
  1. Ultralytics checkpoint 的參數 requires_grad=False，最初載入後未恢復 → 步驟 4–7 第一輪實際上凍結了骨幹/Neck/Head
     （結果移到 `runs/_invalid_frozen_backbones/`）。已修正並重跑。
  2. 雙流的 event 輸入格式與步驟 3 event 模型訓練時不同（未做 BGR→RGB、用 [-1,1] 解碼而非 /255）→ event 骨幹的
     預訓練初始化等於失效。已改為與 Ultralytics 完全一致（翻轉通道、/255、補邊 114），event 單流在我們的評估器上
     重現 51.4/32.9（Ultralytics 51.1/33.1）。20-epoch 的雙流結果（步驟 4–6）是在舊格式下訓練與評估的。
- **分光線條件評估**（`tools/lighting_stats.py`、`tools/eval_conditions.py` → `runs/eval_conditions.md`）：
  test 集以白天為主（12,179 張），偏暗 1,000 張，夜間僅 714 張（主要是 zurich_city_12_a），無嚴重過曝場景。
- **步驟 8 光線自適應融合**（`IllumGate`，`hsf/models/fusion.py`）：在每個尺度用小型 MLP 讀
  RGB 影像的亮度直方圖（16 bins）+ 4×4 平均色 + 兩模態特徵的 GAP，輸出每通道閘門 g = 2·sigmoid(·) ∈ (0, 2)；
  neck = rgb + g·(融合輸出 − rgb)，RGB 分支同理。最後一層初始為 0 → g = 1，起始時等於原本的 add / AF。
  訓練時 15% 影像強制 g = 0（modality dropout），讓 Neck 也學會只吃 RGB，閘門才真的能在白天關掉 event。
  參數 +0.19 M。訓練設定同 lr 1e-3 那組（5 epochs、warmup 1、從步驟 3 單流權重開始）。
  對照：`route_add` / `route_af` = 依亮度硬切換（白天用 RGB 模型、偏暗與夜間用融合模型），不需訓練。
- **與論文數字的差異**：我們的 RGB 基準比論文高約 10 點（75.5 vs 65.6），原因不明（論文未詳述資料
  與評估細節）。因此比較以「我們自己的基準 → 各模組的提升幅度」為準。

## 資料位置
- `D:\DSEC\`：原始資料（約 211 GB）
- `D:\DSEC_processed\`：前處理後的資料
- `E:\claude_Workspace\HSF_fusion\`：程式碼、設定、`runs/` 訓練紀錄

路徑集中在 `configs/paths.yaml`。

## 環境
```bat
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe tools\check_env.py
```
VS Code：`Ctrl+Shift+D` 選設定後按 F5。

## 資料夾結構
```
HSF_fusion/
├─ configs/        路徑與實驗設定
├─ hsf/            主程式（資料、模型、訓練）── 之後的步驟陸續加入
├─ tools/          下載、前處理、檢查用的腳本
├─ runs/           訓練輸出（不進版控）
└─ .vscode/        F5 執行設定
```
