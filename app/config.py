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

# Next.js から直接叩く確認用
CORS_ORIGINS = ["http://localhost:3000"]
