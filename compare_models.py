import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalized_match import normalize
def stats(path):
    if not os.path.exists(path): return None
    d=json.load(open(path)); rows=[v for v in d.values() if 'bleu4' in v]
    n=len(rows)
    if not n: return (0,0,0,0)
    b=sum(r['bleu4'] for r in rows)/n
    e=100*sum(r['exact_match'] for r in rows)/n
    nm=100*sum(1 for r in rows if normalize(r.get('ground_truth',''))==normalize(r.get('predicted','')))/n
    return n,b,e,nm
R=os.path.dirname(os.path.abspath(__file__))+'/results'
print("%-12s %8s %8s %8s %12s" % ("model","n","BLEU-4","exact%","norm-match%"))
for label,p in [("CFG (improved)",R+"/eval_cfg.json"),("Gigahorse TAC",R+"/eval_gh.json")]:
    s=stats(p)
    if s: print("%-12s %8d %8.4f %8.2f %12.2f" % (label,s[0],s[1],s[2],s[3]))
    else: print("%-12s  (no results)" % label)
