from radon_bridge.studies.cohort_cmx_full_host import cmx_full_config_from_none


def test_cmx_full_config_from_no_communication_reference():
    parent={"schema":"factorized_ws_v1","name":"none_seed3416","seed":3416,
        "parents":{"cfp":{"path":"a","sha256":"1"},"oct":{"path":"b","sha256":"2"}},"bridges":[]}
    cfg=cmx_full_config_from_none(parent,64,4)
    assert parent["bridges"]==[]
    assert cfg["name"]=="cmx_full_seed3416"
    assert cfg["seed"]==3416
    assert cfg["parents"]==parent["parents"]
    assert cfg["bridges"]==[{"nodes":["cfp_stage3","oct_stage3"],"family":"cmx_full","alignment_tokens":64,"heads":4}]
