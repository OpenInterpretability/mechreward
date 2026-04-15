"""Feature catalog serialization tests (no SAE required)."""


from mechreward.features.catalog import Feature, FeaturePack, load_pack, save_pack


def test_feature_roundtrip():
    f = Feature(
        feature_id=42,
        name="test_feature",
        description="Activates on the letter A.",
        weight=0.5,
        validated=True,
    )
    d = f.to_dict()
    f2 = Feature.from_dict(d)
    assert f == f2


def test_feature_pack_save_load(tmp_path):
    pack = FeaturePack(
        name="test/pack",
        version="0.0.1",
        model_name="google/gemma-2-2b",
        release="gemma-scope-2b-pt-res-canonical",
        sae_id="layer_12/width_16k/canonical",
        description="Unit test pack",
        features=[
            Feature(feature_id=1, name="a", description="desc a", weight=1.0),
            Feature(feature_id=7, name="b", description="desc b", weight=-0.5),
        ],
    )
    path = tmp_path / "pack.json"
    save_pack(pack, path)

    loaded = load_pack(str(path))
    assert loaded.name == pack.name
    assert len(loaded.features) == 2
    assert loaded.feature_ids() == [1, 7]
    assert loaded.feature_weights() == {1: 1.0, 7: -0.5}


def test_load_bundled_pack():
    """The bundled reasoning_pack should load cleanly."""
    pack = load_pack("gemma-2-9b/reasoning_pack")
    assert pack.model_name == "google/gemma-2-9b"
    assert len(pack.features) >= 1


def test_empty_pack_warning(tmp_path, recwarn):
    pack = FeaturePack(name="empty", features=[])
    path = tmp_path / "empty.json"
    save_pack(pack, path)
    load_pack(str(path))
    assert any("zero features" in str(w.message) for w in recwarn)
