# ============================================================
# GENERAL ERROR-CODE PREDICTION FRAMEWORK
# Set TARGET to any error code. The pipeline:
#   Stage 1: scan fleet, find units with most incidents of TARGET
#   Stage 2: four-test screening -> weighted model -> threshold sweep
#            + control-unit specificity check
# Shown here configured for code 304. Change TARGET to reuse.
# ============================================================
import requests, pandas as pd, numpy as np, time, os
from datetime import datetime, timezone, timedelta
from scipy import stats
from sklearn.metrics import roc_auc_score, mutual_info_score

BASE_URL="https://recycling.cleanplanetchemical.com"
EMAIL="YOUR_EMAIL_HERE"; PASSWORD="YOUR_PASSWORD_HERE"   # set via env var in production

TARGET=304                  # <-- the one knob: any error code
TOP_N_UNITS=4
MIN_EVENTS_PER_UNIT=3
LOOKBACK_DAYS=90            # API retains ~84-89 days; 90 is the practical ceiling
LOOK_FORWARD_HOURS=6
DEDUP_GAP_HOURS=2
DEBOUNCE_MINUTES=3; COOLDOWN_HOURS=2
THRESHOLDS=[10,20,30,40,50,60,70,80,90]
# base sensors screened (static levels; swap to slopes by setting USE_SLOPES=True)
BASE_VARS=["Transducer04","Temp08","Energy","DAC1","Feedback06","LoadAmp"]

# four-test pass thresholds (the methodology)
D_PASS=0.8; P_PASS=0.05; AUC_PASS=0.75; MI_PASS=0.05

r=requests.post(f"{BASE_URL}/api/v1/auth/login",json={"email":EMAIL,"password":PASSWORD}); r.raise_for_status()
H={"Authorization":f"Bearer {r.json()['access_token']}"}; print("Logged in.\n")

def d2u(d): return int(datetime.strptime(d,"%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
END=datetime.now(timezone.utc); START=END-timedelta(days=LOOKBACK_DAYS)

def dedup_ts(ts):
    if len(ts)==0: return np.array([])
    g=pd.Series(ts).sort_values(); gap=g.diff()/3600
    return g[(gap>DEDUP_GAP_HOURS)|gap.isna()].values

# ---------- STAGE 1: locate offender units for TARGET ----------
print("="*70); print(f"STAGE 1: finding units with most code-{TARGET} incidents"); print("="*70)
rr=requests.get(f"{BASE_URL}/api/v1/units",headers=H,timeout=60); rr.raise_for_status()
j=rr.json(); ulist=j if isinstance(j,list) else j.get("units",j.get("data",[]))
unit_ids=sorted({int(u.get("id") or u.get("unit_id")) for u in ulist if (u.get("id") or u.get("unit_id")) is not None})
print(f"Scanning {len(unit_ids)} units...\n")

unit_cnt={}; unit_nm={}
for n,uid in enumerate(unit_ids):
    try:
        resp=requests.get(f"{BASE_URL}/api/v1/units/{uid}",headers=H,timeout=60)
        if resp.status_code!=200: continue
        body=resp.json(); nm=body.get("name") or str(uid); st=body.get("statuses") or []
    except: continue
    if not st: continue
    edf=pd.DataFrame(st)
    if "code" not in edf.columns or "start_timestamp" not in edf.columns: continue
    edf["code"]=pd.to_numeric(edf["code"],errors="coerce"); edf=edf.dropna(subset=["code"]); edf["code"]=edf["code"].astype(int)
    edf["ts"]=pd.to_datetime(edf["start_timestamp"],utc=True,errors="coerce"); edf=edf.dropna(subset=["ts"])
    edf=edf[(edf["ts"]>=pd.Timestamp(START))&(edf["ts"]<=pd.Timestamp(END))]
    e=edf[edf["code"]==TARGET]
    if len(e)==0: continue
    inc=dedup_ts((e["ts"].astype("int64")/1e9).values)
    if len(inc)>0: unit_cnt[uid]=len(inc); unit_nm[uid]=nm
    if (n+1)%25==0: print(f"  ...scanned {n+1}/{len(unit_ids)}")

if not unit_cnt: print(f"\nNo code-{TARGET} incidents found anywhere. Stop."); raise SystemExit
ranked=sorted(unit_cnt.items(),key=lambda x:x[1],reverse=True)
print(f"\nTop units for code-{TARGET}:")
for uid,c in ranked[:15]: print(f"  unit {uid} ({unit_nm.get(uid,'?')}): {c} incidents")

TRAIN_UNITS=dict(list({uid:unit_nm.get(uid,str(uid)) for uid,c in ranked if c>=MIN_EVENTS_PER_UNIT}.items())[:TOP_N_UNITS])
if not TRAIN_UNITS: print(f"\nNo unit has >= {MIN_EVENTS_PER_UNIT} incidents. Stop."); raise SystemExit
print(f"\nTraining units: {TRAIN_UNITS}")
ctrl=[uid for uid in unit_ids if uid not in unit_cnt]
CONTROL_UNITS={ctrl[0]:"control"} if ctrl else {}
print(f"Control unit: {CONTROL_UNITS}\n")

# ---------- data helpers ----------
def pull_day(uid,ds):
    cache=f"/content/fleet{TARGET}"; os.makedirs(cache,exist_ok=True); p=f"{cache}/u{uid}_{ds}.csv"
    if os.path.exists(p) and os.path.getsize(p)>1000:
        try: return pd.read_csv(p,on_bad_lines='skip')
        except: pass
    try:
        resp=requests.get(f"{BASE_URL}/api/v1/units/{uid}/diagnostics/main/main/export-csv",
            headers=H,params={"x_min":d2u(ds),"zoom":3,"high_resolution":"true"},timeout=180); resp.raise_for_status()
        if len(resp.content)<500: return pd.DataFrame()
        open(p,"wb").write(resp.content); return pd.read_csv(p,on_bad_lines='skip')
    except: return pd.DataFrame()

def get_df(uid):
    days=[d.strftime("%Y-%m-%d") for d in pd.date_range(START.date(),END.date(),freq="D")]
    ch=[]
    for d in days:
        c=pull_day(uid,d)
        if len(c)>5 and "cctimestamp" in c.columns: ch.append(c)
        time.sleep(0.2)
    if not ch: return pd.DataFrame()
    df=pd.concat(ch,ignore_index=True).sort_values("cctimestamp").drop_duplicates("cctimestamp").reset_index(drop=True)
    return df[(df["cctimestamp"]>=START.timestamp())&(df["cctimestamp"]<=END.timestamp())].reset_index(drop=True)

def get_events(uid):
    try:
        resp=requests.get(f"{BASE_URL}/api/v1/units/{uid}",headers=H,timeout=60); resp.raise_for_status()
        st=resp.json().get("statuses") or []
    except: return np.array([])
    if not st: return np.array([])
    edf=pd.DataFrame(st); edf["code"]=pd.to_numeric(edf["code"],errors="coerce")
    edf=edf.dropna(subset=["code"]); edf["code"]=edf["code"].astype(int)
    edf["ts"]=pd.to_datetime(edf["start_timestamp"],utc=True,errors="coerce")
    e=edf[edf["code"]==TARGET].dropna(subset=["ts"])
    if len(e)==0: return np.array([])
    inc=dedup_ts((e["ts"].astype("int64")/1e9).values)
    return inc[(inc>=START.timestamp())&(inc<=END.timestamp())]

# ---------- STAGE 2: label, screen, score, sweep ----------
print("="*70); print("STAGE 2: four-test screening + threshold sweep"); print("="*70)
train={}
for uid,nm in TRAIN_UNITS.items():
    df=get_df(uid)
    if df.empty: print(f"  {uid}: no data"); continue
    df["cctimestamp"]=pd.to_numeric(df["cctimestamp"],errors="coerce")
    ev=get_events(uid)
    print(f"  unit {uid} ({nm}): rows={len(df):,}  {TARGET} events={len(ev)}")
    train[uid]=(df,ev)
if not train or sum(len(ev) for _,(_,ev) in train.items())==0:
    print("No usable data/events. Stop."); raise SystemExit

# pooled pre-event vs normal samples
pre={c:[] for c in BASE_VARS}; norm={c:[] for c in BASE_VARS}
rng=np.random.default_rng(0)
for uid,(df,ev) in train.items():
    if len(ev)==0: continue
    t=df["cctimestamp"].values; lab=np.zeros(len(df),dtype=bool)
    for e in ev: lab |= (t>e-LOOK_FORWARD_HOURS*3600)&(t<=e)
    pi=np.where(lab)[0]; ni=np.where(~lab)[0]
    if len(pi)==0: continue
    ns=rng.choice(ni,size=min(len(ni),len(pi)*5),replace=False)
    for c in BASE_VARS:
        if c in df.columns:
            a=pd.to_numeric(df[c],errors="coerce").values
            pre[c].extend(a[pi][~np.isnan(a[pi])]); norm[c].extend(a[ns][~np.isnan(a[ns])])

def cohens_d(a,b):
    a,b=np.asarray(a),np.asarray(b); na,nb=len(a),len(b)
    if na<2 or nb<2: return 0
    sp=np.sqrt(((na-1)*a.std(ddof=1)**2+(nb-1)*b.std(ddof=1)**2)/(na+nb-2))
    return 0 if sp==0 else (a.mean()-b.mean())/sp

screen=[]
for c in BASE_VARS:
    a=np.array(pre[c]); b=np.array(norm[c])
    if len(a)<10 or len(b)<10: continue
    d=cohens_d(a,b)
    try: mw=stats.mannwhitneyu(a,b,alternative="two-sided").pvalue
    except: mw=1.0
    y=np.r_[np.ones(len(a)),np.zeros(len(b))]; x=np.r_[a,b]
    try: auc=roc_auc_score(y,x); auc=max(auc,1-auc)
    except: auc=0.5
    try: xb=pd.qcut(x,q=10,labels=False,duplicates="drop"); mi=mutual_info_score(y,xb)
    except: mi=0
    sc=int(abs(d)>=D_PASS)+int(mw<P_PASS)+int(auc>=AUC_PASS)+int(mi>=MI_PASS)
    screen.append({"var":c,"AUC":round(auc,3),"d":round(d,3),"mw_p":round(mw,4),"MI":round(mi,3),"score":sc})
sdf=pd.DataFrame(screen).sort_values(["score","AUC"],ascending=False).reset_index(drop=True)
print(f"\n=== FOUR-TEST SCREENING (code {TARGET}) ==="); print(sdf.to_string(index=False))

KEY=sdf[sdf["score"]>=3]["var"].tolist()
if len(KEY)<2: KEY=sdf.head(5)["var"].tolist(); print(f"\n(no 3/4 passers; using top-5 by AUC: {KEY})")
else: print(f"\nSelected predictors: {KEY}")

aucmap={r["var"]:r["AUC"] for _,r in sdf.iterrows()}
raw={v:max(aucmap[v]-0.5,0.001) for v in KEY}; tot=sum(raw.values()); W={v:raw[v]/tot for v in KEY}
ZONES={}
for v in KEY:
    a=np.array(pre[v]); b=np.array(norm[v])
    if a.mean()<b.mean(): lo,hi=float(np.min(a)),float(np.percentile(b,25))
    else: lo,hi=float(np.percentile(b,75)),float(np.max(a))
    ZONES[v]={"low":min(lo,hi),"high":max(lo,hi)}
print("\n=== MODEL ===")
for v in KEY: print(f"  {v}: w={W[v]:.3f} zone=[{ZONES[v]['low']:.3f},{ZONES[v]['high']:.3f}]")

def risk_of(df):
    s=np.zeros(len(df))
    for v in KEY:
        if v in df.columns:
            col=pd.to_numeric(df[v],errors="coerce").values
            inz=(col>=ZONES[v]["low"])&(col<=ZONES[v]["high"])&(~np.isnan(col)); s+=inz*W[v]
    return s*100

def detect(ts,risk,thr,deb,cool):
    above=risk>=thr; out=[]; i=0; last=-np.inf; n=len(ts)
    while i<n:
        if not above[i]: i+=1; continue
        s=ts[i]
        while i<n and above[i]: i+=1
        e=ts[i-1]
        if e-s<deb: continue
        if s-last<cool: continue
        out.append(s+deb); last=e
    return out

deb=DEBOUNCE_MINUTES*60; cool=COOLDOWN_HOURS*3600; lf=LOOK_FORWARD_HOURS*3600
recs=[]; incs=[]
for uid,(df,ev) in train.items():
    rk=risk_of(df); t=df["cctimestamp"].values
    for thr in THRESHOLDS:
        for tt in detect(t,rk,thr,deb,cool):
            hit=ev[(ev>=tt)&(ev<=tt+lf)] if len(ev) else np.array([])
            recs.append({"unit":uid,"thr":thr,"tt":tt,"tp":len(hit)>0,"lead":((hit.min()-tt)/60 if len(hit)>0 else np.nan)})
    for e in ev: incs.append({"unit":uid,"ts":e})
tr=pd.DataFrame(recs); ti=pd.DataFrame(incs)

rows=[]
for thr in THRESHOLDS:
    t=tr[tr["thr"]==thr] if len(tr) else pd.DataFrame()
    n=len(t); tp=int(t["tp"].sum()) if n else 0
    if len(ti) and tp>0:
        caught=sum(1 for _,inc in ti.iterrows() if len(t[(t["tp"])&(t["unit"]==inc["unit"])&(t["tt"]>=inc["ts"]-lf)&(t["tt"]<=inc["ts"])])>0)
        rec=caught/len(ti)
    else: rec=0
    prec=tp/n if n>0 else 0
    leads=t[t["tp"]]["lead"].dropna() if n else pd.Series(dtype=float)
    rows.append({"Threshold":f"{thr}%","Triggers":n,"TP":tp,"FP":n-tp,
                 "Recall":f"{rec*100:.1f}%","Precision":f"{prec*100:.1f}%",
                 "MedLead":f"{leads.median():.0f}" if len(leads) else "—"})
res=pd.DataFrame(rows)
print(f"\n{'='*90}\nCODE {TARGET} — THRESHOLD SWEEP  (incidents: {len(ti)})\n{'='*90}")
print(res.to_string(index=False))

print(f"\n{'='*90}\nCONTROL CHECK (zero-{TARGET} unit — should fire FEW)\n{'='*90}")
for uid,nm in CONTROL_UNITS.items():
    df=get_df(uid)
    if df.empty: print(f"  {uid}: no data"); continue
    df["cctimestamp"]=pd.to_numeric(df["cctimestamp"],errors="coerce"); rk=risk_of(df); t=df["cctimestamp"].values
    for thr in [30,40,50]:
        print(f"  control {uid} @ {thr}%: {len(detect(t,rk,thr,deb,cool))} triggers (all false)")

res.to_csv(f"/content/code{TARGET}_sweep.csv",index=False)
sdf.to_csv(f"/content/code{TARGET}_screening.csv",index=False)
print(f"\nSaved. DONE — this is the general framework; change TARGET to apply to any code.")
