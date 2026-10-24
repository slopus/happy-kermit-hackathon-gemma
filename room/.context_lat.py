import json, sys, numpy as np
R=sys.argv[1]
T=[json.loads(l) for l in open(R+'/truth.ndjson')]; K=[json.loads(l) for l in open(R+'/kermit.ndjson')]
S=json.load(open(R+'/score.json'))
offs={e['step']:e['t'] for e in T if e.get('event')=='utt_off'}
rows=[]
for r in S['steps']:
    if 'latency_s' not in r: continue
    t_off=offs[r['step']]; t_k=t_off+r['latency_s']
    te=[e for e in K if e.get('event')=='turn_end' and t_off-0.5<e['t_rx']<t_k]
    tc=[e for e in K if e.get('event')=='turn_candidate' and t_off-0.5<e['t_rx']<t_k]
    hd=[e for e in K if e.get('event')=='heard' and t_off-0.5<e['t_rx']<t_k]
    pl=[e for e in K if e.get('event')=='play' and t_off<e['t_wall']<t_k+0.2]
    f=lambda L: round(L[0]['t_rx']-t_off,2) if L else None
    rows.append((r['step'], r['mode'], f(tc), f(te), f(hd), round(pl[0]['t_wall']-t_off,2) if pl else None, r['latency_s']))
print('step mode  candidate  turn_end  heard(STT)  play  room')
for x in rows: print(*x)
