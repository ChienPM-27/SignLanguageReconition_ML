# ASL Citizen — Full EDA report (tự động)

Nguồn bảng: cache: /kaggle/working/asl_outputs/metrics/full_eda_cache. Tổng 83,399 video, 2,731 gloss, 52 signer, 49.6 GB.

## Tổng quan theo split
```
        videos  glosses  signers     gb  median_vid_per_gloss  median_vid_per_signer
split                                                                               
train  40154.0   2731.0     35.0  24.59                  15.0                  425.0
val    10304.0   2731.0      6.0   6.26                   4.0                 1847.0
test   32941.0   2731.0     11.0  18.76                  12.0                 2998.0
ALL    83399.0   2731.0     52.0  49.60                  31.0                 1496.0
```

## Cân bằng class (video/gloss)
```
        min    p5  median   p95   max   gini  pct_gloss_le10
train   9.0  13.0    15.0  16.5  24.0  0.046           0.696
val     2.0   3.0     4.0   5.0   7.0  0.097         100.000
test    7.0  10.0    12.0  14.0  20.0  0.050           6.298
total  21.0  29.0    31.0  32.0  45.0  0.022           0.000
```

- Median train/gloss = 15; val/gloss = 4; test/gloss = 12.
- Gloss có ≤10 video train: 0.7%. Signer/gloss (train) trung vị = 14.

## Signer
- Top-5 signer = 18.0% video. Video/signer: min 2, median 1496, max 3004.
- Signer/split: {'test': 11, 'train': 35, 'val': 6}.

## Số mẫu/class khi chọn K class (kịch bản tốt nhất)
```
      K  min_train  median_train  min_val  min_test  min_signers_train  total_videos
0     5         13          16.0        6        12                 12           183
1    10         13          13.5        6        12                 12           339
2    20         13          13.0        6        11                 12           649
3    50         12          15.0        5        10                 11          1689
4   100         12          15.0        5         9                 11          3239
5   200         12          14.0        5         9                 11          6265
6   500          9          14.0        4         8                  9         15612
7  1000          9          14.0        4         8                  9         31116
8  2000          9          14.0        3         7                  8         61495
9  2731          9          15.0        2         7                  8         83399
```

## Mẫu video phân tầng
- 1384/1384 video đọc được; fps median 30; độ phân giải phổ biến: (np.int64(640), np.int64(480)); độ dài p50/p90 = 2.74/4.34s.
- MediaPipe thấy tay: 37.5% frame (trung bình).

## Câu hỏi để chọn pipeline (điền sau khi cả nhóm đọc)
1. Dùng bao nhiêu class cho mốc chính? (xem đường cong K)
2. Chọn mô hình/val thế nào khi val chỉ có vài signer? (CV theo signer?)
3. Signer nào/điều kiện quay nào làm MediaPipe kém? Có cần lọc không?
4. Chuẩn hoá & độ dài chuỗi T chọn theo phân bố độ dài ở trên.