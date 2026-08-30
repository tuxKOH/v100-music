#!/usr/bin/env python3
"""Track a continuous fundamental through a fine sweep's tonal candidates."""
from __future__ import annotations
import argparse, csv, math
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser(description='对 sweep 候选做谐波折叠和相邻连续性跟踪')
    p.add_argument('candidates',type=Path); p.add_argument('-o','--output',type=Path,default=Path('sweep_pitch_tracked.csv'))
    p.add_argument('--segments',type=int,default=193); p.add_argument('--min-hz',type=float,default=180); p.add_argument('--max-hz',type=float,default=2500)
    p.add_argument('--min-db',type=float,default=4.0); p.add_argument('--max-jump-oct',type=float,default=.45)
    p.add_argument('--plot',type=Path,default=Path('sweep_pitch_tracked.png')); a=p.parse_args()
    raw=list(csv.DictReader(a.candidates.open()))
    bin20=lambda f: round(float(f)/20)*20
    occ=Counter(bin20(x['frequency_hz']) for x in raw)
    groups={}
    for x in raw:
        f=float(x['frequency_hz']); db=float(x['relative_prominence_db'])
        if f<300 or f>12000 or db<a.min_db or 9500<=f<=10500 or occ[bin20(f)]>0.18*a.segments: continue
        groups.setdefault(int(x['index']),[]).append((f,db))
    states=[]
    for i in range(a.segments):
        bases=[]
        for f,db in groups.get(i,[]):
            for harmonic in range(1,17):
                base=f/harmonic
                if not a.min_hz<=base<=a.max_hz: continue
                # Higher harmonics are useful evidence but less certain.
                score=max(.1,db-3.0)/math.sqrt(harmonic)
                found=None
                for j,(old,oldscore,oldh) in enumerate(bases):
                    if abs(math.log2(base/old))<.018:
                        found=j; break
                if found is None: bases.append((base,score,harmonic))
                elif score>bases[found][1]: bases[found]=(base,score,harmonic)
        bases.sort(key=lambda x:x[1],reverse=True)
        states.append(bases[:80])
    # Ensure every segment has at least a weak state; this keeps the path indexed.
    for i,s in enumerate(states):
        if not s:
            states[i]=[(float('nan'),0.01,0)]
    dp=[]; back=[]
    for i,s in enumerate(states):
        cur=[]; prev=dp[-1] if dp else None
        for j,(freq,score,harm) in enumerate(s):
            emission=math.log1p(score)
            if prev is None: cur.append(emission); continue
            best=(-1e30,0)
            for k,(pf,_,_) in enumerate(states[i-1]):
                if math.isfinite(freq) and math.isfinite(pf):
                    jump=abs(math.log2(freq/pf))
                    cost=18.0*jump*jump + (80.0*max(0.0,jump-a.max_jump_oct)**2)
                else: cost=8.0
                value=prev[k]-cost
                if value>best[0]: best=(value,k)
            cur.append(emission+best[0]);
        dp.append(cur); back.append([] if prev is None else [max(range(len(states[i-1])),key=lambda k: dp[i-1][k]-(18*(abs(math.log2(x[0]/states[i-1][k][0]))**2) if math.isfinite(x[0]) and math.isfinite(states[i-1][k][0]) else 8)) for x in s])
    idx=max(range(len(dp[-1])),key=lambda j:dp[-1][j]); chosen=[]
    for i in range(len(states)-1,-1,-1):
        chosen.append(states[i][idx]);
        if i: idx=back[i][idx]
    chosen.reverse(); rows=[]
    for i,(freq,score,harm) in enumerate(chosen):
        rows.append({'index':i,'size':512+i*8,'tracked_fundamental_hz': '' if not math.isfinite(freq) else round(freq,2),'evidence_score':round(score,2),'harmonic_divisor':harm})
    with a.output.open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    valid=[r for r in rows if r['tracked_fundamental_hz']!='']
    fig,ax=plt.subplots(figsize=(14,7),dpi=160); ax.plot([r['size'] for r in valid],[r['tracked_fundamental_hz'] for r in valid],'.-',ms=3,lw=.8)
    ax.set(xlabel='Matrix size',ylabel='Tracked fundamental candidate (Hz)',title='V100 FP32 v2: continuity-constrained harmonic track',xlim=(500,2060),ylim=(a.min_hz,a.max_hz)); ax.grid(alpha=.25)
    fig.text(.01,.01,'Dynamic path with harmonic folding; low-level/fixed bins excluded. Preliminary, verify by listening.',fontsize=8); fig.tight_layout(rect=(0,.03,1,1)); fig.savefig(a.plot)
    print(f'写入 {a.output} 和 {a.plot}；有效段 {len(valid)}/{len(rows)}')
if __name__=='__main__': main()
