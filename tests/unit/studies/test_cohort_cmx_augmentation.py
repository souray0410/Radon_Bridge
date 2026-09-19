import copy
from radon_bridge.studies.cohort_cmx_augmentation import matched_configs


def test_cmx_matched_configs_share_exact_A_and_add_only_parallel_residual():
    base={
        "schema":"factorized_ws_v1","name":"cmx_frm_seed3416","seed":3416,
        "parents":{"cfp":{"path":"c","sha256":"1"},"oct":{"path":"o","sha256":"2"}},
        "bridges":[{"nodes":["cfp_stage3","oct_stage3"],"family":"cmx_frm","alignment_tokens":64}],
    }
    bases={
        "cfp_stage3":{"path":"cfp.npz","sha256":"a"},
        "oct_stage3":{"path":"oct.npz","sha256":"b"},
    }
    href={"path":"host.json","sha256":"h"}
    cfg=matched_configs(base,bases,href)
    assert set(cfg)=={"host_continue","host_radon","host_linear"}
    assert cfg["host_continue"]["bridges"]==base["bridges"]
    for key,mode in (("host_radon","radon"),("host_linear","linear_resample")):
        assert cfg[key]["bridges"][0]==base["bridges"][0]
        added=cfg[key]["bridges"][1]
        assert added["parallel_to"]==0
        assert added["mode"]==mode
        assert added["basis_files"]==bases
        assert cfg[key]["augmentation_host"]==href
    assert base["bridges"]==[{"nodes":["cfp_stage3","oct_stage3"],"family":"cmx_frm","alignment_tokens":64}]
