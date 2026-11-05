"""パスと閾値をまとめる場所（調整はここだけで済むようにする）"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"
MODEL_NAME = "face_landmarker float16/1"
OBJ_PATH = BASE_DIR / "assets" / "canonical_face_model.obj"
TEMPLATE_DIR = BASE_DIR / "templates"

# アルゴリズムを変えたら上げる（/health で返す）
FEATURES_VERSION = "0.1.0"

# 入力のルール
MAX_BYTES = 10 * 1024 * 1024
MAX_SIDE = 1920

# 顔解析（資料16枚目）
DECIMALS = 4  # 数値は小数4桁に丸める
MIN_FACE_PX = 200  # 顔の幅がこれ未満なら LOW_RESOLUTION
MAX_POSE_DEG = 15.0  # yaw か pitch がこれを超えたら FACE_NOT_FRONTAL
MIN_BRIGHTNESS = 80  # 顔領域の平均 L(0〜255) がこれ未満なら LOW_LIGHT

# 特徴検出 v0（資料18枚目）
NORM_FACE_WIDTH = 600  # 1. 顔幅をこの px にそろえる
NORM_MARGIN = 0.15  # 1. 切り抜きの余白（顔幅に対する割合）
PART_MARGIN_PX = 12  # 2. 眉・唇・鼻の穴を除外するときの余白
EYE_MARGIN_PX = 24  # 2. 目を除外するときの余白（まつげ・目の下の影を避ける）
SKIN_ERODE_PX = 8  # 2. 肌マスクを少し収縮
MEDIAN_KSIZE = 21  # 3. medianBlur の大きさ
DARK_DIFF = 12  # 3. d = median(L) - L がこれ以上
MIN_AREA, MAX_AREA = 4, 400  # 4. 面積(px)
MIN_CIRCULARITY = 0.5  # 4. 円形度
RED_DIFF = 8  # 5. 周囲より a* がこれ以上高ければ赤み（捨てる）
MOLE_MIN_DIAMETER = 6  # 6. 直径(px)がこれ以上
MOLE_MIN_DIFF = 25  # 6. かつ d がこれ以上なら mole
MAX_FEATURES = 30  # 6. 最大件数

# 時代合成 v0（資料19枚目）
SKIN_COLOR_BLEND = 0.6  # 色合わせ: テンプレートの肌に寄せる割合
SKIN_STD_RATIO = (0.5, 2.0)  # 色合わせ: 標準偏差を変える倍率の範囲（陰影を潰しすぎない）
MASK_ERODE_PX = 10  # マスクの収縮
MASK_BLUR_PX = 21  # マスクのぼかし（奇数）
JPEG_QUALITY = 90
# 顔の大きさ合わせ: 利用者の FACE_OVAL をテンプレートの顔の楕円に合わせる
FACE_AXES_PER_EYE = (1.11, 1.41)  # 両目の間隔に対する楕円の半径（横, 縦）。meta に face が無いときに使う
FACE_COVER_MARGIN = 1.06  # 楕円より少し大きくして、収縮・ぼかしで元の顔がはみ出さないようにする
FACE_MAX_STRETCH = 0.1  # 縦横の倍率差の上限（顔の形を崩しすぎない）

# 画面から追加する背景テンプレート
CUSTOM_PREFIX = "custom_"
THUMB_WIDTH = 400

# 同一人物判定（簡易版）: ランドマーク形状の比較。顔認識モデルではない
VERIFY_THRESHOLD = 70.0  # 一致度(%)がこれ以上なら同一人物とみなす（暫定。同じ写真の回転・反転で 70〜96%）
VERIFY_RMS_ZERO = 0.06  # 形状の差(RMS)がこれ以上で一致度 0%（平均顔との差は約 0.1）

# 顔認証: OpenCV の YuNet（顔検出）+ SFace（顔認識）。モデルは tools/download_face_models.py で取得
FACE_DETECTOR_PATH = BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx"
FACE_RECOGNIZER_PATH = BASE_DIR / "models" / "face_recognition_sface_2021dec.onnx"
FACE_DETECTOR_NAME = "yunet 2023mar"
FACE_RECOGNIZER_NAME = "sface 2021dec"
FACE_DB_PATH = BASE_DIR / "data" / "face_db.json"  # 特徴量だけを保存（顔画像は保存しない）
FACE_DETECT_SIDE = 640  # 長辺をこの px に縮小して検出（YuNet は大きく写った顔を見落とすため）
FACE_DETECT_SCORE = 0.8  # 検出の信頼度がこれ未満の顔は無視
FACE_MIN_PX = 80  # 顔の幅（元画像の px）がこれ未満なら LOW_RESOLUTION
FACE_MATCH_THRESHOLD = 0.363  # コサイン類似度がこれ以上で本人とみなす（OpenCV 推奨値。展示環境で要調整）
FACE_ENROLL_IMAGES = 3  # 登録に使う枚数
FACE_IDENTIFY_TOP_K = 3  # identify で返す候補の数
USER_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"

# Next.js から直接叩く確認用
CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]
