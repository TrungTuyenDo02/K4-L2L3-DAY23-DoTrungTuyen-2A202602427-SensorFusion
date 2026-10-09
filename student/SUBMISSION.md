# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Đỗ Trung Tuyến
- MSSV: 2A202602427
- Email: trungdo2002.hvt@gmail.com
- Link repo (fork): https://github.com/TrungTuyenDo02/K4-L2L3-DAY23-DoTrungTuyen-2A202602427-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): <điền sau commit cuối>

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`: `compare`, frames 0–198, segment `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`, seed 0
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: precision 0.9701, recall 0.7004, tp 519 / fp 16 / fn 222
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: rmse 0.1503 m, matches 502, sum_sq_err 11.3436, ghost_track_frames 0, missed_gt_frames 239, mean_confirmed_tracks 2.5226
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: rmse 0.1359 m, matches 502, sum_sq_err 9.2668, ghost_track_frames 0, missed_gt_frames 239, mean_confirmed_tracks 2.5226
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss: hai mode có cùng `matches` = 502, `ghost_track_frames` = 0, `missed_gt_frames` = 239 và cùng `mean_confirmed_tracks`, vì camera chỉ refine trạng thái EKF, không cộng/trừ score và không tạo/xoá track. Vì thế khác biệt nằm hoàn toàn ở sai số: fused 0.1359 m so với lidar 0.1503 m (fused − lidar = −0.0144 m). Camera ở đây là tâm hộp 2D ground-truth + nhiễu seed, nên mức cải thiện này không chứng minh chất lượng perception độc lập với GT.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?

   - **LiDAR:** `z` là vị trí 3×1 (x, y, z) tính bằng mét, `R = diag(0.1², 0.1², 0.1²)`. Phép đo chỉ là đổi hệ trục nên tuyến tính, `H` chính là phần xoay của `veh_to_sens` (`sensors.py: get_H`).
   - **Camera:** `z` là pixel 2×1 (u, v), `R = diag(5², 5²)` px² (`build_camera_measurement`). Phép đo là chiếu pinhole `u = c_i − f_i·y/x`, `v = c_j − f_j·z/x`, **phi tuyến**, nên phải dùng EKF, với `H` là Jacobian (đạo hàm phép chiếu nhân ma trận xoay) tính tại trạng thái dự đoán.
   - Camera không có độ sâu, nên chỉ chỉnh được hướng nhìn chứ không chỉnh được khoảng cách. Đơn vị của `R` cũng khác (m² so với px²).

2. Vì sao cần gating Mahalanobis trước khi gán?

   - `d² = γᵀ S⁻¹ γ` với `S = H P Hᵀ + R` (`association.mahalanobis_distance`), rồi so với ngưỡng χ² mức 0.995 (12.84 cho lidar 3 chiều, 10.60 cho camera 2 chiều) trong `chi2_gate`.
   - Vì chuẩn hoá theo độ bất định nên track có `P` lớn thì cổng rộng, track chắc chắn thì cổng hẹp. Đo ngoài cổng bị loại trước khi ghép tham lam, nên không kéo track đi sai.
   - Khoảng cách Euclid dùng một ngưỡng cố định, không xét `P`, và không so được mét với pixel.

3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.

   - Là **track-then-fuse**. `run_lab.py` dùng một `TrackManager` duy nhất; mỗi frame `KF.predict` một lần, rồi `associate_and_update` với LiDAR (AssocL), rồi với camera (AssocC) trên cùng danh sách track. Dữ liệu thô hai cảm biến không bị gộp trước khi tracking.
   - Trên log: `grade_run.log` có record riêng cho từng `(mode, frame)` với `mode` là `lidar` hoặc `fused`, cùng `det_tp/det_fp/det_fn/valid_gt` vì dùng chung detector.
   - Hai mode có cùng `matches` = 502, `ghosts` = 0, `misses` = 239 và chỉ khác `sum_sq_err` (11.34 so với 9.27). Camera chỉ tinh chỉnh trạng thái của track do LiDAR tạo ra.

4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?

   - Phép chiếu `h(x)` đặt điểm sai chỗ, nên innovation `γ = z − h(x)` lệch **có hệ thống** theo hướng ngang `u` (cùng dấu, không trung bình về 0), và `d²` tăng theo mức lệch.
   - Số liệu ở mục Bonus: lệch yaw 0° → 0.5° thì trung bình |γ_i| tăng 12.9 → 28.5 px, `d²` trung bình 2.25 → 7.64 (ngưỡng gate 10.60).
   - Từ 1° trở lên, `d²` của hầu hết đo vượt ngưỡng nên gate chặn (chỉ còn 20/2799 đo được dùng), fused gần như quay về lidar-only.
   - Với lệch nhỏ (0.25°–0.5°), đo vẫn lọt gate nhưng bị lệch, làm RMSE fused xấu hơn lidar. Vậy gate chặn được lỗi lớn chứ không chặn được lỗi nhỏ.

5. Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng?
   Giải thích vì sao lidar quyết định score/init/delete còn camera chỉ EKF update.

   - Frame rỗng thì `meas_list` rỗng, không suy ra được cảm biến từ danh sách. `manage_tracks(unassigned_tracks, unassigned_meas, sensor)` vẫn phải chạy để trừ điểm track lidar bị bỏ lỡ trong FOV và xoá track hết điểm hoặc `P` quá lớn. Nếu bỏ qua, track ma sẽ tồn tại mãi. Vì vậy `associate_and_update` luôn gọi `manage_tracks` ở cuối, kể cả khi rỗng.
   - Trong `manager.py`, `manage_tracks` và `handle_updated_track` chỉ tác động khi `sensor.name == "lidar"`. Camera ở lab này là tâm hộp 2D ground-truth cộng nhiễu, FOV hẹp và không có độ sâu; nếu camera cũng cộng/trừ điểm thì xe ngoài tầm nhìn camera bị xoá oan. Camera chỉ cập nhật EKF để tinh chỉnh vị trí.

6. Nêu điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track.

   - **Điểm:** track mới có `score = 1/6`. Mỗi lidar hit cộng 1/6 (tối đa 1), mỗi miss trong FOV trừ 1/6.
   - **Xác nhận:** khi `score > 0.8`. Log cho thấy confirmed từ frame 4, sau 4 hit liên tiếp (5/6 ≈ 0.833). Score đúng bằng 0.8 vẫn là `tentative`.
   - **Giữ confirmed:** đã confirmed thì không bị hạ trạng thái; một lần miss chỉ còn 5/6 ≥ 0.6 nên vẫn giữ.
   - **Xoá** (`should_delete_track`): `P[0,0] > 9` hoặc `P[1,1] > 9` (bất kể score); hoặc confirmed mà `score < 0.6`; hoặc chưa confirmed mà `score ≤ 0`.

## Bonus (không bắt buộc)

Liệt kê phần bonus đã làm, file bằng chứng trong `student/bonus/` và kết quả chính
(xem [RUBRIC.md](../RUBRIC.md) mục 2). Không làm thì ghi "Không".

- **Đã làm: Phân tích calibration (+4).** Làm lệch yaw extrinsic camera (xoay quanh trục z của xe) ở 6 mức, đo innovation, số đo camera được dùng và RMSE. Bằng chứng: script `student/bonus/calibration_sweep.py` và số liệu thô `student/bonus/calibration_sweep.json`. Cách làm: chạy detector một lần (frame 0–198, segment như trên, seed 0), lưu detection, rồi chạy lại tracker cho mỗi mức lệch với cùng chuỗi nhiễu camera. Mức 0° tái tạo đúng kết quả chấm điểm (fused 0.1359 m, lidar-only 0.1503 m). Không sửa `platform/` hay `workspace/`; lệch chỉ áp lên đối tượng `Sensor` camera của tracker, pixel vẫn lấy từ nhãn ground-truth đúng.

  | Lệch yaw | RMSE fused (m) | fused − lidar (m) | Đo camera được dùng / tổng | mean \|γ_i\| (px) | mean γ_i (px) | mean d² (cổng 10.60) |
  |---|---|---|---|---|---|---|
  | 0° | 0.1359 | −0.0144 | 509 / 2799 | 12.9 | −11.9 | 2.25 |
  | 0.1° | 0.1458 | −0.0045 | 506 / 2799 | 16.4 | −15.6 | 3.09 |
  | 0.25° | 0.1755 | +0.0252 | 484 / 2799 | 21.4 | −20.5 | 4.71 |
  | 0.5° | 0.2129 | +0.0626 | 304 / 2799 | 28.5 | −27.6 | 7.64 |
  | 1° | 0.1738 | +0.0235 | 20 / 2799 | 84.0 | −73.5 | 3.43 (*) |
  | 2° | 0.1780 | +0.0277 | 19 / 2799 | 88.3 | −80.6 | 3.85 (*) |

  Lidar-only (tham chiếu): RMSE 0.1503 m; `matches` = 502, ghost = 0, missed = 239 ở mọi mức lệch. (*) Với 1° và 2°, `d²` chỉ tính trên số ít đo còn lọt gate nên bị thiên lệch xuống (survivorship).

  Nhận xét: (1) Innovation ngang γ_i lệch cùng dấu (âm) và tăng đều theo mức lệch; ở mức 0° vẫn có |γ_i| ≈ 12.9 px có thể do tâm hộp 2D ground-truth không trùng đúng hình chiếu tâm hộp 3D (giả thuyết, chưa kiểm chứng), nên cần so sánh tương đối với mức 0°. (2) Mức 0.25°–0.5° nguy hiểm nhất: đo vẫn lọt gate (484 và 304 lần cập nhật) nhưng bị lệch, kéo RMSE fused xấu hơn lidar (+0.025 và +0.063 m); mức 0.5° vượt ngưỡng nhất quán 0.05 m. (3) Từ 1° trở lên |γ| vượt xa ngưỡng gate (d² > 10.60) nên gate chặn gần hết đo (còn 19–20 lần), RMSE ổn định quanh 0.174–0.178 m, vẫn kém lidar-only 0.150 m vì còn vài đo lệch lọt qua. (4) Số đo camera "không dùng" gồm cả đo không có track tương ứng (2290/2799 ngay ở mức 0°), nên chỉ so sánh tương đối giữa các mức. Hạn chế: camera là tâm hộp 2D ground-truth cộng nhiễu, chỉ thử lệch yaw, một segment, một seed.

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Claude Code (Claude Sonnet 5.5)
- Dùng cho phần nào (hàm, câu hỏi, debug): cài môi trường; viết code các hàm Part E–H (`kalman.py`, `camera_fusion.py`, `association.py`, `track_management.py`); debug lỗi `float()` trên mảng 1x1 của NumPy 2.5; chạy `fusion-run-lab`; viết bản nháp 6 câu giải thích Part E–H và phần phân tích bonus (script `calibration_sweep.py`, bảng nhận xét); tôi đọc lại, đối chiếu với code/log, chỉnh cách trình bày và chịu trách nhiệm về nội dung.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): `pytest student/tests -q` (128 passed, không còn failed/xfailed); chạy `fusion-run-lab --fusion compare --seed 0` với frame 0–198; đối chiếu `metrics.json` với `grade_run*.log` và `tools/check_submission.py`.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa)
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [ ] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [ ] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [ ] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [ ] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
