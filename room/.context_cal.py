import numpy as np, acoustics as A, noise as N
from acoustics import FS, BLOCK
rirs, meta = A.compute_rirs(); S = A.sources(); names=[s for s in S if S[s]['group']=='noise']
lib = N.Library(0); cal = N.calibrate(lib, rirs, names); print({k: round(v,1) for k,v in cal.items()})
conv = A.StreamConv([[rirs[f"{s}|mic"]] for s in names])
for base in (45, 60, 70):
  nf = N.NoiseField(lib, cal, names, seed=1); nf._random_events = lambda dt: None
  out=[]; per={s:[] for s in names}
  for k in range(int(30*FS/BLOCK)):
    b = nf.block(base); X = np.stack([b[s] for s in names]); Y = conv.process(X)
    out.append(Y[:,0].sum(0))
    for i,s in enumerate(names): per[s].append(Y[i,0])
  y = np.concatenate(out)[FS:]
  print(base, 'total', round(A.digital_to_spl(A.a_weight_db(y)),1), {s: round(A.digital_to_spl(A.a_weight_db(np.concatenate(per[s])[FS:])),1) for s in names})
