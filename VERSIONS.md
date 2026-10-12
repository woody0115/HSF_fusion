# 模型版本總覽

更新：2026-10-06。測試集：DSEC-Det 13 個 test 序列（13,893 張；白天 12,179／偏暗 1,000／夜間 714）。
偵測數字為 mAP50-95（%），評估器同 Ultralytics（conf 0.001、IoU 0.7）。分光線結果來自 `runs/eval_conditions.md`。
各版本權重：GitHub Release `models-2026-10-12`（https://github.com/woody0115/HSF_fusion/releases/tag/models-2026-10-12）。
架構圖、各模組參數量與速度的詳細版：https://claude.ai/code/artifact/fcf8f32a-ac8b-4ec4-bc46-31cc56203e69

## 依用途挑哪個版本

| 用途 | 版本 | 檢查點 |
|---|---|---|
| 整體最準（以白天為主的場景） | RGB 單流 | `runs/step3/rgb_yolov8s/weights/best.pt` |
| 夜間 | add lr1e-3 | `runs/step4/dual_add_lr1e3_e5/best.pt` |
| 偏暗 | AF lr1e-3 | `runs/step5_6/dual_af_lr1e3_e5/best.pt` |
| 融合模型中整體最佳 | AF lr1e-3／iaf_add／iaf_af（都是 53.5） | 見下表 |
| 開放世界，重視找出 unknown | OWOD + event 忽略（做法 3） | `runs/step7/owod_add_evtign/T1..T4.pt` |
| 開放世界，重視已知類別準度 | OWOD 原版 | `runs/step7/owod_add/T1..T4.pt` |

## 一、物件偵測（封閉集合，7 類）

| # | 版本 | 步驟 | 設定 | 全部 | 白天 | 偏暗 | 夜間 | 檢查點 |
|---|---|---|---|---|---|---|---|---|
| 1 | RGB 單流 | 3 | YOLOv8s、COCO 預訓練、10 epochs、lr 0.01、batch 16 | **53.9** | **54.1** | 60.6 | 42.2 | `step3/rgb_yolov8s/weights/best.pt` |
| 2 | Event 單流（GEAR） | 3 | 同上 | 32.9 | 32.9 | 39.0 | 26.7 | `step3/event_yolov8s/weights/best.pt` |
| 3 | add e20 | 4 | 20 epochs、lr 0.005；best = epoch 8 ※ | 51.9 | 51.7 | 59.9 | 44.7 | `step4/dual_add_e20/best.pt` |
| 4 | AF e20 | 5 | 20 epochs、lr 0.005；best = epoch 1 ※ | 52.9 | 53.1 | 60.1 | 41.5 | `step5_6/dual_af_e20/best.pt` |
| 5 | HSF e20（TFS+AF） | 6 | 20 epochs、lr 0.005、4 張片段；best = epoch 16 ※ | 51.5 | 51.6 | 59.0 | 44.1 | `step5_6/dual_hsf_e20/best.pt` |
| 6 | add lr1e-3 | 4 | 5 epochs、lr 0.001、warmup 1 | 53.0 | 53.1 | 61.0 | **45.5** | `step4/dual_add_lr1e3_e5/best.pt` |
| 7 | AF lr1e-3 | 5 | 同 6 | 53.5 | 53.8 | **62.8** | 42.5 | `step5_6/dual_af_lr1e3_e5/best.pt` |
| 8 | HSF lr1e-3 | 6 | 同 6 | 53.1 | 53.2 | 61.2 | 42.5 | `step5_6/dual_hsf_lr1e3_e5/best.pt` |
| 9 | iaf_add（光線閘門） | 8 | 同 6 + 閘門 + 15% modality dropout | 53.5 | 53.7 | 60.9 | 42.5 | `step8_iaf/dual_iaf_add_lr1e3_e5/best.pt` |
| 10 | iaf_af（光線閘門） | 8 | 同 9 | 53.5 | 53.5 | 62.1 | 42.9 | `step8_iaf/dual_iaf_af_lr1e3_e5/best.pt` |

※ 3–5 使用舊的 event 輸入格式（未做 BGR→RGB、[-1,1] 解碼），event 分支等於沒用到預訓練；6 以後是修正後格式。
所有雙流模型的兩個骨幹都從 1、2 初始化。

參考值（不需訓練，`tools/eval_conditions.py` 的 `route_*`）：

| 版本 | 說明 | 全部 | 白天 | 偏暗 | 夜間 |
|---|---|---|---|---|---|
| route_add | 白天用 1，偏暗／夜間用 6 | 53.8 | 54.1 | 60.8 | 45.5 |
| route_af | 白天用 1，偏暗／夜間用 7 | 53.8 | 54.1 | 62.9 | 42.5 |

論文數字（HSF，ICRA 2025）：RGB 44.4、event 24.8、add 43.3、AF 44.5、HSF 45.6。

## 二、開放世界 DSEC-OWOD（Task 1–4）

任務：T1 car → T2 bus、truck → T3 pedestrian、rider → T4 bicycle、motorcycle。融合方式 add。

| # | 版本 | 步驟 | 設定 | 檢查點 |
|---|---|---|---|---|
| 11 | OWOD 原版 | 7 | T1 基礎訓練 10 epochs（COCO 初始化，只用 car），之後凍結、只調嵌入；T1 car mAP50-95 69.6 | `step7/owod_add/` |
| 12 | OWOD + event 忽略（做法 3） | 9 | 同 11，T1 與 object 嵌入訓練時 event 密度前 10% 的背景 anchor 不算負樣本；T1 car 63.9 | `step7/owod_add_evtign/` |

| Task | 版本 | U-Recall ↑ | U-Recall@30 | WI ↓ | A-OSE ↓ | PK | CK | Both |
|---|---|---|---|---|---|---|---|---|
| T1 | 11 | 9.9 | 9.8 | 0.0102 | 3109 | – | 88.0 | 88.0 |
| T1 | 12 | **16.4** | 12.9 | 0.0180 | 4996 | – | 86.4 | 86.4 |
| T2 | 11 | 5.8 | 5.0 | 0.0025 | 1955 | 88.0 | 19.1 | 42.1 |
| T2 | 12 | **12.3** | 9.0 | 0.0055 | 2897 | 86.4 | 16.0 | 39.5 |
| T3 | 11 | 6.7 | 6.5 | 0.0025 | 1327 | 41.9 | 0.2 | 25.3 |
| T3 | 12 | **26.0** | 22.3 | 0.0045 | 1565 | 39.2 | 0.3 | 23.7 |
| T4 | 11 | – | – | 0 | 0 | 25.2 | 3.3 | 19.0 |
| T4 | 12 | – | – | 0 | 0 | 23.7 | 1.0 | 17.2 |

AP 為 AP50（OWOD 慣例）。版本 11 的 T1–T3 U-Recall@30 來自 `runs/step9_evt_unknown/owod_add_rescore.md` 的 α=0 列，
WI、A-OSE 以該檔重新評估的數值為準時會與上表的原始 step 7 數值略有差異（原始評估沒有 U-Recall@k）。

推論時 event 重新評分（做法 1，`tools/owod_event_rescore.py`）只是評估方式，不是新模型：
- 用在版本 11：測試集上 α=1 平均 U-Recall +2.5、α=2 +6.6，但訓練集挑到 α=0.25（+0.44，未過 +0.5 門檻）。
- 用在版本 12：訓練集挑到 α=0，沒有額外好處。
結果：`runs/step9_evt_unknown/*_rescore.md`。

## 三、封存、測試用（不要拿來用）

| 資料夾 | 原因 | 大小 |
|---|---|---|
| `runs/_invalid_frozen_backbones/` | 載入 Ultralytics 權重後 requires_grad 沒打開，骨幹被凍結 | 約 0.85 GB |
| `runs/_invalid_event_input_format/` | event 輸入格式修正前跑到一半的 add lr1e-3 | 125 MB |
| `runs/step3/event_yolov8s_pipetest/` | event 單流的流程測試 | 49 MB |
| `runs/step4/dual_add_smoke/`、`runs/step5_6/dual_*_smoke/` | 雙流流程測試（2 個短 epoch） | 約 0.47 GB |
| `runs/step7/owod_add_smoke/` | OWOD 流程測試 | 378 MB |

以上合計約 1.9 GB，**一律保留不刪**，作為專案歷程的一部分（見 `CHANGELOG.md`）。

## 四、程式版本對應

| 功能 | 開關 | 程式 |
|---|---|---|
| 融合方式 | `train_dual.py --fusion add / af / hsf / iaf_add / iaf_af` | `hsf/models/fusion.py` |
| 光線閘門的 modality dropout | `--modality-drop 0.15` | `hsf/models/dual_yolov8.py` |
| OWOD event 忽略 | `train_owod.py --event-ignore 0.1` | `hsf/owod/event.py`（`EventIgnoreLoss`） |
| OWOD 推論重新評分 | `owod_event_rescore.py --alphas ...` | `hsf/owod/event.py`（`rescore_unknown`） |
| 分光線評估 | `eval_conditions.py --models ...` | `tools/eval_conditions.py` |
