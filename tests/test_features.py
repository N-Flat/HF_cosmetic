"""特徴検出 v0 のテスト: 人工のほくろ・そばかす・ニキビで、見つける・分類する・除外するを確認"""
from tests.conftest import post_image


def find_near(features, p, width, height, tol_px=6):
    return [f for f in features
            if abs(f["xy"][0] * width - p[0]) < tol_px and abs(f["xy"][1] * height - p[1]) < tol_px]


def test_detects_and_classifies(client, spotted_face):
    features = post_image(client, spotted_face["bytes"]).json()["features"]
    for name, kind, p in spotted_face["spots"]:
        near = find_near(features, p, spotted_face["width"], spotted_face["height"])
        if kind is None:
            assert not near, f"{name} は検出されないはず（赤みとして除外）"
        else:
            assert [f["type"] for f in near] == [kind], f"{name}: {near}"


def test_clean_face_has_few_features(client, face_bytes):
    """修整された肌の写真では、まつげや影を拾わない"""
    assert len(post_image(client, face_bytes).json()["features"]) <= 2


def test_mole_fields(client, spotted_face):
    features = post_image(client, spotted_face["bytes"]).json()["features"]
    for f in features:
        assert 0 < f["size"] < 0.05
        assert 0 < f["confidence"] <= 1
