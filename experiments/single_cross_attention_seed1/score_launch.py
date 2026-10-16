import json,os,pathlib,subprocess,sys
r=pathlib.Path(__file__).resolve().parent
cfg=json.loads((r/'benchmark/run.json').read_text());rt=cfg['runtime']
env=dict(os.environ)
env['PYTHONPATH']=os.pathsep.join([str(r/'runner')]+rt.get('score_extra_pythonpath',[])+[rt['anew_source'],rt['cbg_source']])
env['PATH']=str(pathlib.Path(rt['score_python']).parent)+os.pathsep+env.get('PATH','')
for script,args in [(r/'runner/check_environment.py',['score']),(r/'score_resume.py',[]),(r/'compare_results.py',[])]:
 subprocess.run([rt['score_python'],'-u',str(script),*args],env=env,check=True,cwd=r)
