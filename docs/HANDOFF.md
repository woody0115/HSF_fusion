# 專案交接：HSF_fusion（Event + RGB 融合偵測，延伸到開放世界）

- **整理日期**：2026-11-02。實驗與文件的最後更新是 2026-10-15。
- **Repo**：`E:\claude_Workspace\HSF_fusion`，GitHub 私人 repo `woody0115/HSF_fusion`，分支 `main`，最新 commit `bf6cffb`。
- **本文的用途**：開新對話時先讀這份。數字以 `VERSIONS.md`、`runs/` 的結果檔為準。
- **標記**：🔴 必須記住、⚠️ 有陷阱或限制、❓ 尚未確定。

---

## 1. 三十秒總覽

- **目標**：融合 event camera 和 RGB 做自駕車物件偵測。先重現 HSF 論文，再延伸到開放世界 Task 1–4（YOLO-UniOW 的做法）。
- **已完成**：封閉集 #1–#10、開放世界 #11–#12、CK 崩潰診斷、驗證集切分、續訓、OVD／OWOD 序列切分、Unknown AP、12 頁規劃簡報。
- 🔴 **目前最大的問題**：開放世界的新類別學不起來（T3 CK 只有 0.2–0.4）。原因是 Neck 只學過 car，框回歸也只學過 car，只調文字嵌入救不回來。
- 🔴 **下一步**：實作 OVD 階段（偵測器 + Text Encoder LoRA），在 30 個 OWOD 序列上重跑 T1–T4，和 #11、#12 比較。**OVD 的程式還沒寫。**
- ❓ **卡住的地方**：OVD 階段要用哪些類別的標註（有洩漏問題），要學長確認。本機沒有學長回覆的紀錄。

---

## 2. 工作規則（使用者要求）

- 🔴 **說明一律用繁體中文，程式註解用英文。**
- 🔴 **不刪任何舊版本和封存資料**，包括 `runs/_invalid_*`、`*_smoke`、`*_pipetest`。
- 🔴 **每個修正與改良都記進 `CHANGELOG.md`**（最新在最上面）和架構文件的歷程分頁。
- 🔴 `camera_url.txt` 含攝影機密碼：不複製、不分享、不上傳。IP cam 的部分不進 GitHub。
- 不輸入密碼或 token，不改系統安全設定。
- 右相機的 event 要保留（之後可能做立體）。
- commit 和 push 要等使用者開口。

---

## 3. 基礎流程

```
DSEC-Det 原始資料 (D:)
  → 前處理：YOLO 標註 + GEAR event 影像 (E: SSD)
  → 單流基準 #1 RGB、#2 event
  → 雙流融合 add / AF / HSF（骨幹從 #1、#2 初始化）
  → 光線閘門 iaf（我們的延伸）
  → 開放世界 #11（COCO 初始化，T1 只用 car，之後凍結、只調嵌入）
  → event 參與 unknown 判斷 #12
  → CK 崩潰診斷
  → 下一步：OVD 微調 + OWOD 重跑
```

| 步驟 | 內容 | 結果 |
|---|---|---|
| 0–2 | 下載（約 211 GB）、環境、GEAR 與標註前處理 | 70,379 張有標註影格、390,118 個框 |
| 3 | 單流基準 | RGB 53.9、event 32.9 |
| 4–6 | add、AF、HSF | lr 1e-3 版 53.0、53.5、53.1 |
| 7 | DSEC-OWOD（#11） | T3 U-Recall 6.7、CK 0.2 |
| 8 | 光線閘門（#9、#10） | 都是 53.5，閘門沒學會看光線 |
| 9 | event 參與 unknown（#12） | T3 U-Recall 26.0 |
| 10 | CK 崩潰診斷 | H1 + H3 同時成立，H2 排除 |
| 之後 | 驗證集、續訓、OVD 切分、U-AP、簡報、demo | 完成 |

### 開放世界流程細節（步驟 7）

1. **模型**：雙流 YOLOv8s。🔴 **RGB 和 event 兩條骨幹都從 COCO 的 `weights/yolov8s.pt` 初始化**，不是從步驟 3–6 的權重。
2. **T1 基礎訓練**：全部可訓練，只用 car 標註，10 epochs、lr 0.01、batch 8。
3. **調 object 萬用嵌入**（只看過 car）。
4. **每個 Task**：模型全部凍結，只學新類別嵌入和 unknown 嵌入（目標是 object 分數）。舊類別嵌入凍結，所以不需要 replay。
5. **Task 切分**：T1 car → T2 bus、truck → T3 pedestrian、rider → T4 bicycle、motorcycle。

---

## 4. 論文與參考資料

| 文件 | 位置 | 重點 |
|---|---|---|
| **HSF**：*Hierarchical Spatiotemporal Fusion for Event-Visible Object Detection*（Jhong et al., ICRA 2025） | `C:\Users\woody\Downloads\Hierarchical_Spatiotemporal_Fusion_for_Event-Visible_Object_Detection.pdf` | 重現目標。**沒有公開程式碼**，依描述自行實作。單流 lr 0.01、batch 16、10 epochs；完整模型 lr 0.005、batch 8、40 epochs |
| **YOLO-UniOW**：*Efficient Universal Open-World Object Detection*（Liu et al.，arXiv 2412.20645） | `C:\Users\woody\OneDrive\文件\2412.20645v1.pdf` | OWOD 方法、wildcard、文字編碼器 LoRA。4.3 節有實作設定 |
| **OWOD Survey**（Li et al.，arXiv 2410.11301） | `C:\Users\woody\OneDrive\文件\2410.11301v2.pdf` | 指標定義 |
| **10/01 HISLAB 會議紀錄** | `C:\Users\woody\.claude\uploads\0514b0c7-8b3c-4f30-a42c-3f8630beb215\da7c8a8b-2026-10-01_HISLAB___transcribed_on_12-Oct-2026_21-20-11.pdf` | 學長的規劃。逐字稿 txt 當時是貼進對話的，本機沒有找到檔案 |

### 論文數字對照（mAP50 / mAP50-95）

| 模型 | 論文 | 我們 |
|---|---|---|
| RGB | 65.6 / 44.4 | **75.6 / 53.9** |
| Event | 41.9 / 24.8 | 51.4 / 32.9 |
| add | 64.6 / 43.3 | – / 53.0 |
| AF | 65.7 / 44.5 | – / 53.5 |
| HSF | 67.2 / 45.6 | – / 53.1 |

- ⚠️ **我們的基準比論文高約 10 點，原因不明**，所以只比「相對基準的提升」。
- ❓ 未驗證的假設：論文可能用 8 類平均，我們用 7 類（train 類沒有框）。75.6 × 7/8 = 66.2，接近論文的 65.6。
- ⚠️ 趨勢也不同：論文的 HSF 最好，我們 AF 最好，而且融合都沒贏 RGB。

### 10/01 會議：學長講清楚的事

1. 🔴 **要做簡報**，先把規劃做好；擔心方向做錯。期中考是 10/26 那週。
2. 用 Event + RGB 的行車資料集跑；預期效果不好，**原因在 YOLO 本身，不是 event camera**。
3. UniOW 先微調 Text Encoder 再做 OWOD，所以**我們也要微調**，Text Encoder 用 **LoRA**。
4. 先統計影像數，**以影片為單位切**：例如 50 部拿 10–20 部做 OVD，約 30 部做 wildcard learning。
5. 也可以找其他 Event + RGB 行車資料集，很自由。
6. Task 怎麼切都可以，也可以少於四個。
7. 🔴 **指標：PK、CK、Both，再加 Unknown AP**；T1 沒有 PK。

---

## 5. 核心概念

### 資料

- **DSEC-Det**：60 個序列、70,379 張（訓練 47 序列 56,486 張、測試 13 序列 13,893 張）。
- 8 類標籤，但 train 沒有框，實際 7 類：pedestrian、rider、car、bus、truck、bicycle、motorcycle。
- ⚠️ **類別極不平衡**：car 228,199 個框（佔 77%），rider 只有 2,727。pedestrian 約 80% 小於 32²。
- 測試集光線：白天 12,179、偏暗 1,000、夜間 714（⚠️ 夜間幾乎來自同一個序列）。

| 切分 | 序列 | 影格 | 用途 |
|---|---|---|---|
| ovd | 12 | 13,128 | OVD 微調 |
| owod | 30 | 37,945 | T1–T4 |
| val | 5 | 5,413 | 挑權重（含夜間序列 zurich_city_09_e） |
| test | 13 | 13,893 | 只報告最終數字 |

定義在 `configs/splits.yaml`。OVD 的夜間序列是 zurich_city_09_a。

### GEAR（event 表示法）

- 每張 RGB 影像之間約 50 ms 的 event，累積成 3 通道影像 `[h, feat, num]`，c=8、τ=10。
- `feat` 是這段時間的極性累積，`num` 是事件數，`h` 是長期記憶（忙碌的像素遺忘歷史）。
- 🔴 **event 輸入格式**：現行格式（10/05 16:13 起）是 BGR→RGB、÷255、補邊 114。舊格式沒翻通道，用 [−1,1] 解碼。

### 模型

- **雙流 YOLOv8s**：兩條骨幹（第 0–9 層），在第 4、6、9 層（P3／P4／P5，128／256／512 通道）融合，融合結果回饋給兩條分支。Neck 和偵測頭來自 RGB 模型。
- **add**：逐元素相加。16.2 M 參數。
- **AF**：自適應混合池化到 8×8 → 1 層 transformer → 上採樣加回分支 → 串接 + 1×1 Conv。三處 λ 閘門初始為 0，所以起始等於 add。21.2 M 參數。
- **TFS**：event 分支上的卷積 GRU，閘門同時看 event 和 RGB。
- **HSF** = TFS + AF。
- **IllumGate（iaf，我們的延伸）**：亮度直方圖、4×4 色、特徵 GAP 進 MLP，g = 2·sigmoid ∈ (0,2)，加 15% modality dropout。
- **`ContrastCls`（對比式分類頭）**：`logit = exp(s)·⟨e, T/|T|⟩ + b`，e 是 anchor 的 512 維嵌入，T 是 `ClassTable` 的文字嵌入。第 0–7 列是類別，第 8 列 object，第 9 列 unknown。
- **文字嵌入**：CLIP ViT-B/32 文字端，已合併 UniOW 的 LoRA，存在 `configs/owod_text_embeds.npz`。**目前是固定的，沒有微調。**

### 指標

| 指標 | 意思 | 方向 |
|---|---|---|
| PK | 之前學過的類別的 AP50 | 越高越好 |
| CK | 這個 Task 新學類別的 AP50 | 越高越好 |
| Both | 全部已知類別的 AP50 | 越高越好 |
| U-Recall | unknown 物體被 unknown 預測框到的比例（IoU 0.5） | 越高越好 |
| U-AP | unknown 預測對 unknown 標註的 AP50（**新增**，學長要求） | 越高越好 |
| WI | 開放世界對已知類別精度的影響（已知 recall 0.8） | 越低越好 |
| A-OSE | unknown 被誤判成已知的總數 | 越低越好 |

- ⚠️ 封閉集用 mAP50-95（conf 0.001、IoU 0.7）；開放世界用 AP50（每張最多 100 框）。不要混著比。
- ⚠️ 我們的 A-OSE 用 score ≥ 0，和其他論文不能直接比。
- mAP 或召回率的差距要寫「**點**」，不要寫「%」。

### 診斷假設

- **H1 特徵問題**：凍結的 Neck 分不出新類別。✅ 成立。
- **H2 訓練量不足**：❌ 排除。
- **H3 框回歸問題**：框回歸只學過 car。✅ 成立。

### 方法：event 參與 unknown

- **做法 1（推論重新評分）**：依框內 event 密度調整 unknown 分數，只是評估方式，`tools/owod_event_rescore.py`。
- **做法 3（`EventIgnoreLoss`）**：T1 和 object 嵌入訓練時，event 密度前 10%（每張各自算，`q=0.1`）、又不在已知框內的背景 anchor，不當分類負樣本。**不是自監督。** 這就是 #12。

---

## 6. 檔案地圖

```
HSF_fusion/
├─ VERSIONS.md                版本總覽（所有數字）
├─ CHANGELOG.md               完整歷程，最新在最上面
├─ README.md                  進度與設計決定
├─ configs/
│  ├─ paths.yaml              資料路徑
│  ├─ splits.yaml             val／ovd 序列
│  ├─ owod_tasks.yaml         T1–T4
│  └─ owod_text_embeds.npz    固定的文字嵌入
├─ docs/
│  ├─ HANDOFF.md              本文件
│  ├─ answers_for_planning_agent.md   診斷細節、C 類設計問題、E 節 13 項問題
│  └─ presentation_fixes.md   24 頁簡報的逐頁修正清單
├─ hsf/
│  ├─ engine.py               train、train_temporal、resume、final_test
│  ├─ data/dataset.py         資料集與 split_filter
│  ├─ data/gear.py            GEAR
│  ├─ data/event_sim.py       從畫面模擬 event（demo 用）
│  ├─ models/dual_yolov8.py   雙流（FUSE_AT = 4, 6, 9）
│  ├─ models/fusion.py        add、AF、TFS、HSF、IllumGate
│  ├─ owod/model.py           ContrastCls、ClassTable、to_open_world
│  ├─ owod/train.py           tune_*、evaluate_owod
│  ├─ owod/metrics.py         OWOD 指標（含 U-AP）
│  └─ owod/event.py           EventIgnoreLoss、rescore_unknown
├─ tools/
│  ├─ preprocess_dsec.py      前處理
│  ├─ train_single.py／train_dual.py／train_owod.py   訓練
│  ├─ eval_conditions.py      分光線評估
│  ├─ reeval_owod.py          重新評估 OWOD（含 U-AP）
│  ├─ diag_ck.py              B1–B4、gate
│  ├─ diag_data_stats.py      B5、B6
│  ├─ select_ovd_split.py     挑 OVD 序列
│  ├─ make_text_embeds.py     產生文字嵌入（要用 YOLO_UniOW_test 的 venv）
│  ├─ owod_event_rescore.py   做法 1
│  ├─ demo_screen.py          螢幕影片 demo
│  ├─ pack_release.py         打包權重
│  └─ run_queue.ps1           訓練排程
└─ runs/
   ├─ step3、step4、step5_6、step7、step8_iaf、step9_evt_unknown
   ├─ diag/                   診斷結果 JSON、owod_reeval.md
   ├─ eval_conditions.md      分光線結果
   └─ _invalid_*              無效實驗，保留不刪
```

### repo 以外的位置

| 內容 | 位置 |
|---|---|
| 原始資料 | `D:\DSEC` |
| 訓練資料（SSD） | `E:\DSEC_train` |
| 權重打包 | `D:\HSF_release`；GitHub Release `models-2026-10-12`（12 個 zip） |
| UniOW 文字編碼器（LoRA 已合併） | `E:\claude_Workspace\YOLO_UniOW_test\models\text_model_s\` |
| UniOW 原始程式與 LoRA 未合併權重 | WSL Ubuntu-24.04，`~/YOLO-UniOW`，conda env `uniow`；`yolo_world/models/backbones/mm_backbone.py` |

---

## 7. 版本與結果

### 封閉集（mAP50-95，7 類）

| # | 版本 | 全部 | 白天 | 偏暗 | 夜間 |
|---|---|---|---|---|---|
| 1 | RGB | **53.9** | **54.1** | 60.6 | 42.2 |
| 2 | Event | 32.9 | 32.9 | 39.0 | 26.7 |
| 6 | add lr1e-3 | 53.0 | 53.1 | 61.0 | **45.5** |
| 7 | AF lr1e-3 | 53.5 | 53.8 | **62.8** | 42.5 |
| 8 | HSF lr1e-3 | 53.1 | 53.2 | 61.2 | 42.5 |
| 9 | iaf_add | 53.5 | 53.7 | 60.9 | 42.5 |
| 10 | iaf_af | 53.5 | 53.5 | 62.1 | 42.9 |

- 🔴 **融合整體沒贏 RGB**，但暗處贏：夜間 add +3.3 點、偏暗 AF +2.2 點。
- #3–#5 是舊 event 格式的 e20 版，不要拿來比較。
- 閘門 g ≈ 0.89 幾乎是常數（每張影像標準差 0.0002）。
- 硬切換 `route_*` 是 53.8，仍輸 RGB，因為兩個模型的信心分數標準不同。

### 開放世界（AP50）

| Task | 版本 | U-Recall | U-AP | WI | A-OSE | PK | CK | Both |
|---|---|---|---|---|---|---|---|---|
| T1 | 11 | 9.9 | – | 0.0102 | 3109 | – | 88.0 | 88.0 |
| T1 | 12 | **16.4** | 0.42 | 0.0180 | 4996 | – | 86.4 | 86.4 |
| T2 | 11 | 5.8 | – | 0.0025 | 1955 | 88.0 | 19.1 | 42.1 |
| T2 | 12 | **12.3** | 0.15 | 0.0055 | 2897 | 86.4 | 16.0 | 39.5 |
| T3 | 11 | 6.7 | – | 0.0025 | 1327 | 41.9 | 0.2 | 25.3 |
| T3 | 12 | **26.0** | 0.20 | 0.0045 | 1565 | 39.2 | 0.3 | 23.7 |
| T4 | 11 | – | – | 0 | 0 | 25.2 | 3.3 | 19.0 |
| T4 | 12 | – | – | 0 | 0 | 23.7 | 1.0 | 17.2 |

- 🔴 **#12 的代價**：U-Recall 提高，但 car 少 1.6 點，WI、A-OSE 都變差。U-AP 仍然很低，表示 unknown 的排序很差。
- 🔴 **#11 用舊 event 格式訓練**（10/05 14:45 完成，早於 16:13 的修正）。上表是它當時的評估。用現行程式重新評估時輸入和訓練不一致，數字會變（T1 car 88.0 → 86.4；U-AP 0.32／0.05／0.01 僅供參考）。#12 的重新評估完全一致。

### 診斷結果（`runs/diag/`）

| 診斷 | 結果 | 結論 |
|---|---|---|
| **B1 線性探測**（#12，7 類完整標註訓練 3 epochs） | 只放開分類頭 → Neck 一起訓練：bicycle 5.47 → 51.54、pedestrian 0.59 → 52.46、7 類平均 21.30 → 55.65 | H1 成立。Neck 那組同時開了框的 loss |
| **B2 只調嵌入加量** | 訓練量或 lr 加 10 倍，T3 CK 最高 0.54（同格式基準 0.43） | H2 排除 |
| **B3 Oracle 框召回**（#12，IoU ≥ 0.5） | car 97.2、truck 95.3、bicycle 90.5、motorcycle 87.1、bus 86.7、rider 63.4、**pedestrian 27.4** | H3 成立 |
| **B4 嵌入空間** | pedestrian、rider、bicycle 對所有嵌入列都是負 cosine；彼此平均特徵 0.83–0.93 | 文字嵌入方向不對，類別也互相混淆 |
| **gate** | 閘門幾乎不看輸入 | 光線自適應沒成功 |

- 🔴 **解讀**：bicycle 是 H1 的乾淨證據（框沒問題，分類頭救不回來）。pedestrian 同時混了 H1 和 H3，不能單獨當 H1 的證據。
- ⚠️ B2、B4 只在 #11 上做過，有格式落差。B1、B3 已在 #12 確認。

### Demo

- `tools/demo_screen.py`：框選螢幕區域跑模型，`--model af|add|iaf_add|rgb|owod`。
- ⚠️ 沒有 event camera，event 是模擬的。AF 用模擬 event 52.9、真實 53.5、沒有 event 52.5，所以 demo 只展示運作方式，不能展示 event 的好處。

---

## 8. 踩過的坑（已修正）

| 問題 | 修正 |
|---|---|
| 🔴 Ultralytics checkpoint 參數 `requires_grad=False`，第一輪只訓練到融合模組 | `load_ultralytics` 打開；舊結果封存到 `_invalid_frozen_backbones` |
| 🔴 雙流 event 輸入格式和單流訓練時不同 | 改成和 Ultralytics 一致；⚠️ 但 #3–#5、#11 是舊格式 |
| AF 初版讓 Neck 收到隨機特徵，mAP 從 0 重學 | 加初始為 0 的 λ 閘門 |
| e20 最佳 epoch 太早（AF 在第 1 個） | 改 lr 1e-3、5 epochs |
| Ultralytics NMS 會原地改寫輸入 | 先 clone 再 NMS |
| `quantile` 遇 NaN 全部失效 | 改 `nanquantile` |
| α 用 car 召回率挑，已飽和 | 改用訓練集 unknown 召回 |
| α 清單整數與浮點 key 不一致而中斷 | 一律轉 float |
| Windows 從 stdin 跑 DataLoader 會卡死 | 測試寫成檔案再跑 |
| Smart App Control 擋 torch DLL | 使用者自行關閉 |
| 重開機中斷訓練 | 每個 epoch 存 `resume.pt`，`--resume` 續訓 |
| 用測試集挑權重 | `--split val` 為預設；`--split legacy` 重現 #1–#12 |
| HDD 隨機讀取太慢 | 訓練資料搬到 E 槽 NVMe |

---

## 9. 已知問題與盲區（第三方審查）

1. 🔴 **unknown 其實看過**：COCO 權重學過 person、bicycle、motorcycle、bus、truck；CLIP／UniOW 的文字編碼器也看過。U-Recall 測的不是「沒見過的物體」。可改用 ImageNet 分類預訓練。
2. 🔴 **event 骨幹用 RGB 的 COCO 權重初始化**，開放世界時它只能從 car 標註學會看 event。
3. 🔴 **#1–#12 都用測試集挑權重**。10/13 起的新實驗才改用驗證集。
4. **只跑一個種子、沒有信賴區間**：0.3–0.9 點的差距可能都是雜訊。
5. **沒有 RGB-only 的 OWOD 基準**：目前無法證明 event 對開放世界有幫助。
6. **沒有做重新載入一致性檢查**。
7. **靜止物體沒有 event**，做法 3 只救得到會動的物體。
8. **OVD 的洩漏風險**：OVD 序列如果用全部 7 類標註，unknown 等於看過。
9. **git 從 10/12 才開始**，步驟 0–9 沒有對應 commit。

### 改善方案（細節見 `docs/answers_for_planning_agent.md` E 節）

| 問題 | 方法 |
|---|---|
| Neck 只學過 car | T2–T4 放開 Neck，加蒸餾和每 Task 50–100 張 replay |
| 框回歸只學過 car | 調新類別時打開框與 DFL loss，或用 event 偽框、OVD 預訓練 |
| 小物體 | 提高輸入解析度到 800–960、加 P2、裁切放大 |
| 嵌入方向相反 | 用視覺特徵的類別平均初始化，放開 logit_scale |
| 類別不平衡 | repeat factor sampling |
| unknown 排序差 | event 忽略改成權重減半、只作用在離已知物體遠處；或加不分類別的物件性頭 |
| object 嵌入只看過 car | 每個 Task 重新調 object |
| 閘門不看輸入 | 製造失效情境的增強、用單流 loss 當輔助監督、改空間閘門 |
| 靜止物體 | 拉長 GEAR 時間窗或多時間窗 |

---

## 10. 下一步

### OVD 實作計畫（🔴 主線）

1. **環境**：專案 venv 裝 `transformers`、`peft`。
2. **模型**：雙流 YOLOv8s + `ContrastCls`，但類別嵌入改成每一步由 Text Encoder + LoRA 算出。
3. **LoRA 設定**（UniOW）：`q_proj`、`k_proj`、`v_proj`、`out_proj`，rank 16、alpha 16、dropout 0.25。
4. **詞彙**：7 個類別名稱、`object`，加行車場景負類別（traffic light、traffic sign、pole、tree、building），避免 LoRA 學成 7 個固定向量。
5. **訓練**：`ovd` 切分，AdamW lr 5e-4、weight decay 0.025，10–20 epochs，`--split val` 挑權重。
6. **匯出**：新的文字嵌入 npz 和偵測器權重。
7. **OWOD**：在 `owod` 切分從 OVD 權重開始跑 T1–T4，模型凍結，只做 wildcard learning。
8. **評估**：test 上算 PK、CK、Both、U-AP、U-Recall，和 #11、#12 比較。
9. **對照組**：同樣的 OVD 訓練，但文字端完全凍結，才能分開 LoRA 本身的貢獻。

- 工作量估計：程式約 1 天，OVD 訓練加對照組 2–3 小時，OWOD 重跑加評估約 3 小時。
- 🔴 **預期**：進步主要會來自影像端（Neck 看到所有類別），不是文字端。
- ⚠️ 🔴 **LoRA 起點，先前有兩種說法**：答案文件說從 WSL 的未合併權重開始，OVD 說明說在已合併的 `text_model_s` 上再加一組 LoRA。**建議用 WSL 未合併權重**，這樣基底和 LoRA 分得開，也和 UniOW 一致。
- 速度參考：雙流 add 約 11 分鐘／epoch，完整 OWOD 流程約 2.5 小時。

### ❓ 需要學長或使用者決定

| 代號 | 問題 | 建議 |
|---|---|---|
| C3 | OVD 用哪些標註：(a) DSEC 全部 7 類（有洩漏）、(b) 只用 T1 類別、(c) 外部資料 | 照學長切法的話是 (a)，報告要註明；乾淨的做法是 (b) 或 (c) |
| – | 只微調 Text Encoder，還是偵測器一起訓練 | 一起訓練，和 UniOW 相同 |
| C4 | LoRA 位置與 rank | 沿用 UniOW 設定 |
| C1 | PK 容許掉多少 | ≤ 2 點 |
| C2 | 要不要 replay | 每 Task 50–100 張 |
| C7 | 成功標準 | T3 CK > 20、PK 掉 ≤ 2 點、T3 U-Recall ≥ 26.0 |
| – | 「實驗室自己論文」的資料集是否就是 DSEC／HSF | 要確認 |

### 其他可選工作

| 項目 | 成本 |
|---|---|
| 修 #11 格式：加舊格式開關 | 約 1 小時 |
| 修 #11 格式：用新格式重訓 | 約 2.5 小時 |
| 加 8 類平均，驗證論文差距的假設 | 小 |
| 跑 RGB-only 的 OWOD 基準 | 約 2.5 小時 |
| 修 24 頁簡報（需要它的 HTML） | 依 `docs/presentation_fixes.md` |
| 在簡報加限制頁 | 內容在 `docs/presentation_fixes.md` 第四節 |

### 現況中要注意的事

- ⚠️ **工作目錄有一個未 commit 的修改**：`hsf/owod/model.py` 把 `model.names` 兩行併成一行，只影響格式，不影響功能。原因不明，保持原樣。
- ⚠️ **E 槽只剩 20 GB**，D 槽 196 GB。每次訓練都會存權重，開跑前要確認空間。
- ⚠️ **文件有幾處過時**：
  - `README.md` 第 87 行寫 `D:\DSEC_processed`，實際訓練資料在 `E:\DSEC_train`（以 `configs/paths.yaml` 為準）。
  - `README.md` 的步驟 10 還寫「進行中」，驗證的說明也還是測試集。
  - `docs/answers_for_planning_agent.md` 的 A1 寫「本機沒有會議記錄」，C3 也寫學長意思不確定；後來已拿到會議紀錄，以本文第 4 節為準。
- 目前沒有訓練在跑（2026-11-02 確認）。
- ❓ 本機沒有紀錄顯示規劃簡報是否已報告，也沒有學長的回覆。

---

## 11. 常用指令

都從 `E:\claude_Workspace\HSF_fusion` 執行：

```bat
.venv\Scripts\python.exe tools\train_dual.py --fusion af --epochs 5 --lr 0.001 --warmup 1 --split val
.venv\Scripts\python.exe tools\train_owod.py --event-ignore 0.1 --split val --resume
.venv\Scripts\python.exe tools\reeval_owod.py
.venv\Scripts\python.exe tools\diag_ck.py b1 --run owod_add_evtign
.venv\Scripts\python.exe tools\demo_screen.py --source screen --model af
```

- 硬體：RTX 5070 12 GB，Python 3.13，Ultralytics 8.3.158（不改原始碼）。
- gh CLI 已安裝並登入 `woody0115`。
- WSL 跑 UniOW 時要設 `PYTHONPATH=.`、`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`。

---

## 12. 線上文件

| 內容 | 連結 |
|---|---|
| GitHub repo | https://github.com/woody0115/HSF_fusion |
| 權重 Release | https://github.com/woody0115/HSF_fusion/releases/tag/models-2026-10-12 |
| 版本與架構文件（含歷程分頁） | https://claude.ai/code/artifact/fcf8f32a-ac8b-4ec4-bc46-31cc56203e69 |
| 第一份報告 | https://claude.ai/code/artifact/80f69c3d-d8dc-49e3-8573-05c3042e496f |
| 12 頁規劃簡報 | https://claude.ai/artifact/JVk7AmPHcMZgcBgFG4xXZ2 |

完整對話紀錄：`C:\Users\woody\.claude\projects\E--claude-Workspace\0514b0c7-8b3c-4f30-a42c-3f8630beb215.jsonl`
