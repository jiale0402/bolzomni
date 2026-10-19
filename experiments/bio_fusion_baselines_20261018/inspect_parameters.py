"""Count effective trainable parameters; no model loading or training."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['enzy','ses'],required=True)
    a=p.parse_args()
    here=Path(__file__).resolve().parent
    code=here/'training_code'/a.method
    sys.path.insert(0,str(code))
    spec=importlib.util.spec_from_file_location('matched_fusion',code/'fusion.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    config=json.loads((here/'configs'/a.method/'seed1.json').read_text())
    fusion=module.Fusion(**config['fusion_kwargs'])
    trainable=sum(p.numel() for p in fusion.parameters() if p.requires_grad)
    total=sum(p.numel() for p in fusion.parameters())
    target=config['parameter_match_target']
    relative_error=trainable/target-1
    assert trainable==config['trainable_parameters']
    assert total==config['total_fusion_parameters']
    assert abs(relative_error)<=config['parameter_match_relative_tolerance']
    print(json.dumps(dict(method=a.method,trainable_parameters=trainable,total_parameters=total,
        frozen_unused_parameters=total-trainable,target=target,relative_error=relative_error,
        widths=config['fusion_kwargs'],matched=True),indent=2))


if __name__=='__main__':main()
