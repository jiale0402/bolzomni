"""Fixed validation reduction used by the training entry point."""
from learning import materialize
from native_checks import fixed_rng,evaluate

def validation_eval(model,epsilon,groups):
    results={}
    for i,(modality,rows) in enumerate(groups.items()):
        with fixed_rng(6729000+i):selected=materialize(rows)
        results[modality]=evaluate(model,epsilon,selected,7729000+i*10000)
    # Same generated-block reduction as the native combined validation objective.
    denominator=sum(v['generated_residues'] for v in results.values())
    results['combined']={k:sum(v[k]*v['generated_residues'] for v in results.values())/denominator for k in ['H','X','total']}
    return results


