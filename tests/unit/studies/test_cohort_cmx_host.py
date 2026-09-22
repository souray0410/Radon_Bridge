from radon_bridge.studies.cohort_cmx_host import cmx_config_from_none


def test_cmx_config_from_no_communication_reference():
    parent={"schema":"factorized_ws_v1","name":"none_seed3416","seed":3416,
        "parents":{"cfp":{"path":"a","sha256":"1"},"oct":{"path":"b","sha256":"2"}},"bridges":[]}
    cfg=cmx_config_from_none(parent,64)
    assert parent["bridges"]==[]
    assert cfg["name"]=="cmx_frm_seed3416"
    assert cfg["seed"]==3416
    assert cfg["parents"]==parent["parents"]
    assert cfg["bridges"]==[{"nodes":["cfp_stage3","oct_stage3"],"family":"cmx_frm","alignment_tokens":64}]
