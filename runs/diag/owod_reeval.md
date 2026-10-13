# DSEC-OWOD re-evaluation with U-AP (test split, AP50, max 100 detections per image)

| run | Task | U-AP ↑ | U-Recall ↑ | WI ↓ | A-OSE ↓ | PK | CK | Both |
|---|---|---|---|---|---|---|---|---|
| owod_add | T1 | 0.32 | 9.8 | 0.0154 | 3017 | - | 86.4 | 86.4 |
| owod_add | T2 | 0.05 | 5.1 | 0.0038 | 1751 | 86.4 | 17.4 | 40.4 |
| owod_add | T3 | 0.01 | 6.5 | 0.0038 | 1710 | 40.3 | 0.4 | 24.4 |
| owod_add | T4 | - | - | 0.0000 | 0 | 24.3 | 2.4 | 18.1 |
| owod_add_evtign | T1 | 0.42 | 16.4 | 0.0180 | 4996 | - | 86.4 | 86.4 |
| owod_add_evtign | T2 | 0.15 | 12.3 | 0.0055 | 2897 | 86.4 | 16.0 | 39.5 |
| owod_add_evtign | T3 | 0.20 | 26.0 | 0.0045 | 1565 | 39.2 | 0.3 | 23.7 |
| owod_add_evtign | T4 | - | - | 0.0000 | 0 | 23.7 | 1.0 | 17.2 |
