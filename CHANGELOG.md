# 專案歷程（修正與改良紀錄）

從頭到現在的每個里程碑、修正與改良，最新的在最上面。所有舊版本與封存資料夾一律保留，不刪除：
無效的實驗移到 `runs/_invalid_*`，測試用的留在 `*_smoke`、`*_pipetest`。
版本編號 #1–#12 對應 `VERSIONS.md`。時間取自排程記錄與檔案時間，標「約」的是推估。

| 日期時間 | 步驟 | 類型 | 內容 | 結果／影響 |
|---|---|---|---|---|
| 2026-10-15 | – | 整理 | 核對另一個 agent 製作的 24 頁簡報（`HSF_OWOD_Weekly_Presentation`），寫成逐頁修正清單 `docs/presentation_fixes.md`；切分統計存成 `runs/diag/ovd_split_stats.txt`、`runs/diag/split_lighting.txt` 以便引用 | 第二版程式碼頁已改成真實程式（剩數字缺位、行號等小問題）；數字與結論頁大多未改，第 11 頁 WI 方向相反 |
| 2026-10-14 | – | 改良 | 螢幕影片 demo `tools/demo_screen.py`（和一開始的 YOLO-UniOW demo 同樣操作：框選螢幕區域、conf／IoU／unknown 滑桿；模型 af／add／iaf_add／rgb／owod）；沒有 event camera，event 由 `hsf/data/event_sim.py` 從畫面模擬（對數亮度變化超過門檻 C），再用同一套 GEAR 編碼；啟動方式：VS Code「Demo: 螢幕影片…」或 `tools/run_demo_screen.bat` | 白天、夜間片段都能框到車；開放世界模式 unknown 分數約 0.1，預設門檻改 0.1 |
| 2026-10-14 | – | 結果 | 模擬 event 的可用性（`tools/event_sim_check.py`，`runs/diag/event_sim_check*.json`）：AF 模型在 DSEC 測試集，真實 event 53.53、模擬 C=0.1 52.94、C=0.07（密度最接近真實）52.76、C=0.05 52.34、沒有 event 52.54；只用 RGB 的 #1 是 53.9 | 模擬 event 只比沒有 event 好 0.4，仍低於只用 RGB；demo 只能展示融合模型怎麼運作，不能展示 event 的好處。demo 預設 C=0.1 |
| 2026-10-14 | 10 | 結果 | B1 在格式一致的 #12 上重做（`runs/diag/b1_linear_probe_owod_add_evtign.json`）；依審查修正解讀：特徵（H1）和框（H3）兩個問題都存在，H1 的乾淨證據是 bicycle | #12：只放開分類頭時 bicycle 5.5、pedestrian 0.6；Neck 一起訓練後 51.5、52.5。和 #11 一致。簡報第 8 頁與答案文件已更新 |
| 2026-10-13 晚 | – | 修正 | 依使用者要求全面複查：(1) 簡報切分頁的夜間比例改為逐張計算（OWOD 15.8→16.7%、驗證 15.1→10.2%）；(2) 答案文件 B2 的比較基準改成同格式重新評估的 #11（CK 0.43），PK 落差的原因更正為格式問題；(3) 做法 1 在 #11 上的 α 掃描（10/06）同樣有輸入格式落差，VERSIONS、答案文件已註明；(4) 兩份 docs 文件補上 #11 舊格式的說明；(5) B1 在格式一致的 #12 上重跑確認 | 結論不變；被影響的數字都已標註 |
| 2026-10-13 晚 | – | 整理 | 依 10/01 會議逐字稿做規劃簡報（12 頁：資料統計、OVD／OWOD 切分、Task 切分、評估指標、目前結果與診斷、OVD 微調流程與設定、待學長確認事項、時程） | 簡報為私人連結，分享需由使用者從頁面 Share 選單開啟 |
| 2026-10-13 晚 | 7 | 修正 | 發現 #11 是用舊的 event 輸入格式訓練（10/05 14:45 完成，早於 16:13 的修正），先前 VERSIONS 只標了 #3–#5 | VERSIONS、答案文件已補註；今天在 #11 上的 B1、B2、B4 有格式落差，建議在 #12 上重做 B1 確認 |
| 2026-10-13 晚 | 7 | 改良 | 評估加上 Unknown AP（U-AP，`hsf/owod/metrics.py`）；`tools/reeval_owod.py` 重新評估 #11、#12 | #12 的 U-AP：T1 0.42、T2 0.15、T3 0.20（unknown 排序很差）；#12 其他指標與原始評估完全一致 |
| 2026-10-13 晚 | – | 改良 | 依學長建議以影片切 OVD 微調資料：`tools/select_ovd_split.py` 從 42 個序列挑 12 個（13,128 張，25.7%），其餘 30 個做 OWOD；`configs/splits.yaml` 新增 `ovd_sequences`，資料集新增 `ovd`、`owod` 切分 | 每類約有 24–28% 的框在 OVD 端（最低 rider 23.7%），兩邊都有夜間序列 |
| 2026-10-13 | 10 | 改良 | 驗證集切分：`configs/splits.yaml` 從 47 個訓練序列切出 5 個（5,413 張，含 1 個夜間序列）；`train_dual.py`／`train_owod.py` 預設 `--split val`（在驗證集挑權重，結束後在測試集評估一次，寫 `test_metrics.json`），`--split legacy` 重現 #1–#12 | 之後的實驗不再用測試集挑模型 |
| 2026-10-13 | 10 | 改良 | 續訓：每個 epoch 存 `resume.pt`（原始權重、EMA、optimizer、scaler、進度），`--resume` 接續；OWOD 已完成的 Task 直接沿用 | 小規模測試通過（`runs/_tests/`）；重開機不必從頭重跑 |
| 2026-10-13 | 10 | 結果 | CK 崩潰診斷（`tools/diag_ck.py`、`tools/diag_data_stats.py`，結果在 `runs/diag/`）：B1 線性探測、B2 調嵌入加訓練量、B3 框召回、B4 嵌入空間、B5／B6 資料統計、D3 閘門逐張分布 | 主因是凍結的 Neck（只放開分類頭時 pedestrian 0.7，放開 Neck 後 51.1）；pedestrian 的框回歸也壞（oracle IoU≥0.5 只有 28.9%）；加訓練量無效；閘門每張影像的標準差 0.0002 |
| 2026-10-13 | – | 整理 | 回覆另一個 agent 的疑問清單：`docs/answers_for_planning_agent.md`（含 13 項問題與改善方案）；`tools/make_text_embeds.py` 把一次性指令存成腳本（已驗證完全重現 npz）；README 進度表更新 | – |
| 2026-10-12 | – | 里程碑 | 12 個有效版本的權重上傳到 GitHub Release `models-2026-10-12`（每版一個 zip，含權重、結果檔、載入說明；`tools/pack_release.py`） | 約 1.3 GB；打包檔在 `D:\HSF_release` |
| 2026-10-12 | – | 里程碑 | 上傳到 GitHub 私人 repo [woody0115/HSF_fusion](https://github.com/woody0115/HSF_fusion)：程式、文件、設定與 87 個文字結果檔 | 權重（.pt）、.venv、圖片、資料集不上傳；本機檔案全部保留 |
| 2026-10-06 21:00 | – | 整理 | 版本與架構解析文件、`VERSIONS.md`、本歷程；決定所有舊版本不刪 | 12 個有效版本、6 個封存資料夾全部保留 |
| 2026-10-06 11:37 | 9 | 結果 | #12 OWOD + event 忽略完成；做法 1 疊加在 #12 上 | U-Recall T1 9.9→16.4、T2 5.8→12.3、T3 6.7→26.0；已知類別 AP50 −1.3～−2.6；疊加做法 1 無額外好處 |
| 2026-10-06 08:58 | 9 | 修正 | `owod_event_rescore.py`：α 清單的 0 是整數，總結時查不到 '0.0' 而中斷 | 改成一律轉成浮點數；做法 1 正式判定 +0.44（門檻 +0.5）→ effect False |
| 2026-10-06 08:54 | 9 | 決定 | 做法 3 啟動時讀到手動測試留下的判定檔（effect True）而開始訓練 | 測試集掃描效果明顯（α=2 平均 +6.6），保留執行並在報告中註明 |
| 2026-10-06 08:43 | – | 修正 | 排程不保存每一步的輸出，失敗時看不到原因 | `run_queue.ps1` 改為每步輸出存到 `runs/queue_logs/<名稱>.log` |
| 2026-10-06 04:12 | 8 | 結果 | #9 iaf_add、#10 iaf_af 分光線評估 | 兩者整體 53.5；閘門值在白天／偏暗／夜間都約 0.89，沒學會看光線；add 加閘門後夜間 45.5→42.5 |
| 2026-10-06 03:08 | 8 | 環境 | 電腦於 03:07 重開機，iaf_af 跑到第 1 個 epoch 一半中斷 | iaf_af 從頭重跑，其餘排程重新接上 |
| 2026-10-06 約 02:30 | 9 | 修正 | 測試發現：Ultralytics NMS 會原地改寫輸入框，多個 α 共用同一份輸出時框被改壞 | 每個 α 先複製一份再做 NMS |
| 2026-10-06 約 02:30 | 9 | 修正 | event 忽略遮罩用 quantile，遇到補邊區的 NaN 結果全是 NaN，一個都沒忽略 | 改用 nanquantile，每批約忽略 7% anchor |
| 2026-10-06 約 02:30 | 9 | 修正 | α 原本用「car 召回率」挑，但已飽和在 96–98%，永遠選 0 | 改用訓練集的 unknown 標註挑 α（同 ORE 做法，不碰測試集） |
| 2026-10-06 約 02:20 | – | 環境 | Windows 上從標準輸入執行含 DataLoader 子行程的腳本會卡死 | 測試腳本一律寫成檔案再執行 |
| 2026-10-06 02:13 | 9 | 改良 | event 參與 unknown 判斷：`hsf/owod/event.py`（event 密度、`EventIgnoreLoss`、`rescore_unknown`）；評估加 U-Recall@10／@30 | 做法 1 推論重新評分、做法 3 訓練時不壓 event 活躍背景 |
| 2026-10-06 01:54 | 8 | 參考 | 硬切換：白天用 RGB、暗處用融合模型 | 整體 53.8，仍輸 RGB 53.9：兩模型信心分數標準不同，混合排名被打亂 |
| 2026-10-06 01:48 | 8 | 改良 | 光線閘門 `IllumGate`（亮度直方圖 + 4×4 色 + 特徵 GAP → 每通道 g∈(0,2)）＋ 15% modality dropout | 初始化時輸出與原融合完全相同（差 0.0）；+0.19 M 參數 |
| 2026-10-05 21:59 | 4–6 | 結果 | lr 1e-3 重訓的 #6–#8 分光線評估 | 融合在暗處贏 RGB：夜間 add +3.3、偏暗 AF +2.2；白天略輸；整體最佳 AF 53.5 < RGB 53.9 |
| 2026-10-05 16:13 | 4–6 | 修正 | 雙流的 event 輸入沒做 BGR→RGB，且用 [−1,1] 解碼而非 ÷255，和 event 單流訓練時不同 | event 預訓練原本等於失效；改為與 Ultralytics 一致，event 單流在自家評估器重現 51.4／32.9；跑到一半的 add 封存到 `_invalid_event_input_format` |
| 2026-10-05 15:48 | 4–6 | 改良 | 低學習率重訓（lr 1e-3、5 epochs、warmup 1）＋ 分光線評估工具（`lighting_stats.py`、`eval_conditions.py`） | e20 的最佳 epoch 太早（AF 在第 1 個），判斷為過擬合 |
| 2026-10-05 14:45 | 7 | 結果 | #11 DSEC-OWOD 完成 | T1–T3 U-Recall 9.9／5.8／6.7；舊類別完全不遺忘；新類別 CK 很差 |
| 2026-10-05 12:13 | 6 | 結果 | #5 HSF e20 | 51.5（最佳 epoch 16） |
| 2026-10-05 05:57 | 5 | 結果 | #4 AF e20 | 52.9（最佳 epoch 1） |
| 2026-10-05 01:51 | – | 環境 | Smart App Control 擋下 torch 的 c10.dll，01:36 排程三步全部失敗 | 使用者自行關閉 Smart App Control 後重啟排程 |
| 2026-10-04 19:43 | 4 | 結果 | #3 add e20 | 51.9（最佳 epoch 8） |
| 2026-10-04 15:51 | 4–7 | 修正 | Ultralytics checkpoint 的參數 requires_grad=False，載入後沒打開 → 第一輪步驟 4–7 只訓練到融合模組 | `load_ultralytics` 改為除 DFL 外全部可訓練；第一輪封存到 `_invalid_frozen_backbones`，全部重跑 |
| 2026-10-04 約 05:00 | 5 | 改良 | AF 加上初始為 0 的 λ 閘門（λe、λv、λout；TFS 的 λt 同理） | 初版 AF 讓 Neck 一開始收到隨機特徵、mAP 從 0 重學；改後起始等於 add |
| 2026-10-04 05:23 | 3 | 結果 | #2 event 單流 | 32.9（論文 24.8） |
| 2026-10-04 約 04:40 | 7 | 改良 | OWOD 程式：對比式分類頭、object／unknown 萬用嵌入、OWOD 指標（已單元測試） | 採 UniOW 高效流程（T1 後凍結、只調嵌入） |
| 2026-10-04 03:30 | 0 | 里程碑 | DSEC-Det 下載與解壓完成（約 211 GB，D 槽） | – |
| 2026-10-04 00:24 | 3 | 結果 | #1 RGB 單流 | 53.9（論文 44.4）；確認評估器與 Ultralytics 差 < 0.3 點 |
| 2026-10-03 約 22:50 | 2 | 改良 | 訓練資料從 D 槽（傳統硬碟）搬到 E 槽 NVMe SSD | 大量小檔隨機讀取不再卡速度 |
| 2026-10-03 22:12 | 2 | 里程碑 | 前處理：YOLO 標註（框裁切、< 2 px 丟棄）、GEAR event 影像 | 70,379 張有標註影格、390,118 個框 |
| 2026-10-03 20:58 | 2 | 里程碑 | GEAR 實作（c=8、τ=10、tanh 前 ÷255、PNG [h, feat, num]） | – |
| 2026-10-03 19:04 | 0–1 | 環境 | DSEC 單連線只有約 1 MB/s → 4 連線 HTTP range 續傳下載器；失真 RGB zip 沒有時間戳 → 用 range 讀取只抽時間戳檔 | 下載速度提高到可接受 |
| 2026-10-03 19:04 | 1 | 環境 | MinGit 的 https 載入失敗 → PATH 前加 ucrt64\bin；pandas 3.0.6 被 Smart App Control 擋 → 固定 2.2.3 | 環境可用 |
| 2026-10-03 19:04 | 1 | 里程碑 | 建立 HSF_fusion 專案（E 槽程式、D 槽大檔），目標：重現 HSF 並延伸到開放世界 Task 1–4 | – |
| 2026-09-30 | 前置 | 里程碑 | YOLO-UniOW：WSL2 跑通、即時 demo（webcam／IP cam／螢幕）、Windows VS Code 版打包、上傳 GitHub（不含 IP cam）、LVIS minival AP | 作為 OWOD 方法與文字嵌入的來源 |
