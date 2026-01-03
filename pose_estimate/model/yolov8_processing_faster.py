import cv2
import numpy as np


def preprocess_image(img: np.ndarray, input_size=(640, 640)):
    """画像前処理、簡素化したletterbox処理
    Params:
        img(np.ndarray): RGB画像
        input_size(tuple):入力サイズ
    Returns:
        np.ndarray
    """
    if img.shape[:2] == (640, 640):
        # 入力画像がインプットサイズならリサイズはパス
        padded_img = img
        r = 1
    else:
        input_w, input_h = input_size
        if len(img.shape) == 3:
            padded_img = np.ones((input_w, input_h, 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114
        r = min(input_w / img.shape[0], input_h / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    # (H, W, C)-> (C, H, W)
    padded_img = padded_img.transpose((2, 0, 1))
    padded_img = np.ascontiguousarray(padded_img)
    return padded_img


# 描画用の色リスト
COLOR_LIST = list(
    [
        [128, 255, 0],
        [255, 128, 50],
        [128, 0, 255],
        [255, 255, 0],
        [255, 102, 255],
        [255, 51, 255],
        [51, 153, 255],
        [255, 153, 153],
        [255, 51, 51],
        [153, 255, 153],
        [51, 255, 51],
        [0, 255, 0],
        [255, 0, 51],
        [153, 0, 153],
        [51, 0, 51],
        [0, 0, 0],
        [0, 102, 255],
        [0, 51, 255],
        [0, 153, 255],
        [0, 153, 153],
    ]
)


def xywh2xyxy(box: np.ndarray) -> np.ndarray:
    """検出ボックスのxywh形式をxyxyに変更"""
    box_xyxy = box.copy()
    box_xyxy[..., 0] = box[..., 0] - box[..., 2] / 2
    box_xyxy[..., 1] = box[..., 1] - box[..., 3] / 2
    box_xyxy[..., 2] = box[..., 0] + box[..., 2] / 2
    box_xyxy[..., 3] = box[..., 1] + box[..., 3] / 2
    return box_xyxy


def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IOU計算
    box,boxesのフォーマットは[x1, y1, x2, y2]
    Params:
        box(np.ndarray): 単一のボックス
        boxes(np.ndarray):  重複するボックス群
    Returns:
        float:IOU
    """
    # 重複領域
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])
    inter_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # 全体領域
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - inter_area

    return inter_area / union_area


def nms_process(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """NMS処理
    Params:
        boxes(np.ndarray):検出ボックス群
        scores(np.ndarray):各ボックスのスコア
        iou_thr(float):iou閾値
    Returns:
        list of int: 残すインデックス
    """
    sorted_idx = np.argsort(scores)[::-1]
    keep_idx = []
    while sorted_idx.size > 0:
        idx = sorted_idx[0]
        keep_idx.append(idx)
        ious = compute_iou(boxes[idx, :], boxes[sorted_idx[1:], :])
        rest_idx = np.where(ious < iou_thr)[0]
        sorted_idx = sorted_idx[rest_idx + 1]
    return keep_idx


def image_to_tensor(image: np.ndarray):
    """画像を入力テンソルに変換、正規化しバッチ形式に
    Params:
      img (np.ndarray): 前処理済み画像 CHW RGB
    Returns:
      input_tensor (np.ndarray): input tensor in NCHW format with float32 values in [0, 1] range
    """
    input_tensor = image.astype(np.float32)  # uint8 to fp32
    input_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0

    # 単一バッチ化
    if input_tensor.ndim == 3:
        input_tensor = np.expand_dims(input_tensor, 0)
    return input_tensor


def postprocess(
    pred_boxes: np.ndarray,
    input_hw: tuple[int, int],
    orig_img: np.ndarray,
    box_score=0.25,
    kpt_score=0.5,
    nms_thr=0.45,
) -> dict:
    """推論データの後処理
    Params:
        pred_boxes(np.ndarray):モデル出力データ
        input_hw(Tuple of int):モデル入力画像の(高さ、幅)
        orig_img(np.ndarray):元画像hwc
        box_score(float):検出閾値
        kpt_score(float):キーポイントの検出閾値
        nms_thr(float):iou閾値
    Returns:
        list of Dict:
    """
    results = []
    input_h, input_w = input_hw
    ratio = min(input_w / orig_img.shape[0], input_h / orig_img.shape[1])
    for output in pred_boxes:
        predict = output.T
        predict = predict[predict[:, 4] > box_score, :]
        scores = predict[:, 4]
        boxes = predict[:, 0:4] / ratio
        boxes = xywh2xyxy(boxes)
        kpts = predict[:, 5:]
        for i in range(kpts.shape[0]):
            for j in range(kpts.shape[1] // 3):
                if kpts[i, 3 * j + 2] < kpt_score:
                    kpts[i, 3 * j : 3 * (j + 1)] = [-1, -1, -1]
                else:
                    kpts[i, 3 * j] /= ratio
                    kpts[i, 3 * j + 1] /= ratio
        idxes = nms_process(boxes, scores, nms_thr)
        result = {
            "boxes": boxes[idxes, :].astype(int).tolist(),
            "kpts": kpts[idxes, :].astype(float).tolist(),
            "scores": scores[idxes].tolist(),
        }
        results.append(result)
    return results


def draw_results(
    result: dict, img: np.ndarray, label_map: dict, with_label=False
) -> np.ndarray:
    """骨格点の描画
    Params:
        result(Dict):検出ボックスと骨格座標、スコア{'boxes':[] ,'kpts':[] ,'scores':[] }
        label_map(Dict):クラス番号とそのクラス名 {0:"person"}
        with_label(bool):ラベルを描画するかどうか
    """
    boxes, kpts, scores = result["boxes"], result["kpts"], result["scores"]
    for box, kpt, score in zip(boxes, kpts, scores):
        x1, y1, x2, y2 = box

        cv2.rectangle(img, (x1, y1), (x2, y2), (200, 60, 200), 1)
        if with_label:
            # ラベル名とスコア描画
            label_str = label_map[0] + f"{score * 100:.0f}%"
            label_size, baseline = cv2.getTextSize(
                label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(
                img,
                (x1, y1),
                (x1 + label_size[0], y1 + label_size[1] + baseline),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                img,
                label_str,
                (x1, y1 + label_size[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        for idx in range(len(kpt) // 3):
            # 骨格点を描画
            x, y, score = kpt[3 * idx : 3 * (idx + 1)]
            if score > 0:
                cv2.circle(img, (int(x), int(y)), 5, COLOR_LIST[idx], -1)

    return img
