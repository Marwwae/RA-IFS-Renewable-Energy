from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================================================================
# RA-IFS-TOPSIS | SINGLE-FILE RESEARCH PIPELINE
# ================================================================
# Run only this file:
#     python main.py
#
# The script contains the complete reproducible dataset, source
# metadata, IFS construction, entropy weighting, standard IF-TOPSIS,
# proposed reliability-adjusted IF-TOPSIS, ablation, range-semantics
# sensitivity, hesitation sensitivity, tables and figures.
# No CSV/Excel/model modules are required to run the analysis.
#
# Research core:
#   published evidence range [L,U]
#       -> normalized interval [l,u]
#       -> intuitionistic fuzzy number
#          mu = l
#          nu = 1-u
#          pi = u-l
#       -> criterion hesitation H_j = mean_i(pi_ij)
#       -> evidence reliability R_j = 1-H_j
#       -> adjusted criterion weight
#          w*_j = w_j R_j / sum_k(w_k R_k)
#       -> standard IF-TOPSIS with baseline weights
#       -> proposed RA-IF-TOPSIS with adjusted weights
#
# IMPORTANT INTERPRETATION:
# - Published ranges are NOT assumed to be confidence intervals.
# - IRENA economic/technical ranges retain their native published semantics.
# - Lifecycle GHG uses the public NREL LCA Harmonization dataset, which
#   provides Q1/median/Q3 quartiles for total life-cycle emissions.
#   This provides a homogeneous quartile-based evidence range across
#   the five technologies used in this case study.
# - The interval-induced IFS mapping is a proposed representation, not a
#   probability model: mu=l, nu=1-u, pi=u-l.
# - The baseline reliability factor is parameter-free: R_j=1-H_j.
# - This is a methodological proof-of-concept case study, not a
#   population-level statistical inference study.
# ================================================================

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)

ALTERNATIVES = ['Solar PV', 'Onshore Wind', 'Hydropower', 'Geothermal', 'Bioenergy']
CRITERIA = ['LCOE', 'Total Installed Cost', 'Capacity Factor', 'Life-cycle GHG']
DIRECTIONS = {'LCOE':'cost', 'Total Installed Cost':'cost', 'Capacity Factor':'benefit', 'Life-cycle GHG':'cost'}

# ----------------------------------------------------------------
# Data used in the paper. Central values are the 2024 weighted averages
# reported by IRENA. Bounds are published evidence bounds retained with
# their original semantics.
# GHG baseline uses NREL Q1-median-Q3 quartile evidence ranges.
# ----------------------------------------------------------------
DATA_ROWS = [
    # Economic/technical evidence: IRENA Renewable Power Generation Costs in 2024
    ['Solar PV','LCOE','cost',0.043,0.032,0.122,'USD/kWh','5th-95th percentile (2024 projects)','IRENA (2025), Renewable Power Generation Costs in 2024'],
    ['Solar PV','Total Installed Cost','cost',691,489,1610,'USD/kW','5th-95th percentile (2024 projects)','IRENA (2025), Renewable Power Generation Costs in 2024'],
    ['Solar PV','Capacity Factor','benefit',17.4,11.5,22.6,'%','5th-95th percentile (2024 projects)','IRENA (2025), Renewable Power Generation Costs in 2024'],
    ['Solar PV','Life-cycle GHG','cost',57,44,73,'gCO2e/kWh','Q1-median-Q3; photovoltaic crystalline-silicon/all technologies','NREL public LCA Harmonization dataset (2021; DOI 10.7799/1819907)'],

    ['Onshore Wind','LCOE','cost',0.034,0.024,0.075,'USD/kWh','5th-95th percentile (2024 projects)','IRENA (2025), Renewable Power Generation Costs in 2024'],
    ['Onshore Wind','Total Installed Cost','cost',1041,727,2110,'USD/kW','5th-95th percentile (2024 projects)','IRENA (2025), Renewable Power Generation Costs in 2024'],
    ['Onshore Wind','Capacity Factor','benefit',34,24,56,'%','5th-95th percentile (2024 projects)','IRENA (2025), Renewable Power Generation Costs in 2024'],
    ['Onshore Wind','Life-cycle GHG','cost',12,7.9,21,'gCO2e/kWh','Q1-median-Q3; land-based wind','NREL public LCA Harmonization dataset (2021; DOI 10.7799/1819907)'],

    ['Hydropower','LCOE','cost',0.057,0.040,0.147,'USD/kWh','reported 2024/regional evidence envelope','IRENA (2025), Renewable Power Generation Costs in 2024'],
    ['Hydropower','Total Installed Cost','cost',2267,1000,7460,'USD/kW','reported project/country evidence envelope','IRENA (2025), Renewable Power Generation Costs in 2024'],
    ['Hydropower','Capacity Factor','benefit',48,32,67,'%','5th-95th percentile (2024 large hydropower)','IRENA (2025), Renewable Power Generation Costs in 2024'],
    ['Hydropower','Life-cycle GHG','cost',21,8.4,27,'gCO2e/kWh','Q1-median-Q3; hydropower/all technologies','NREL public LCA Harmonization dataset (2021; DOI 10.7799/1819907)'],

    ['Geothermal','LCOE','cost',0.060,0.033,0.090,'USD/kWh','reported 2024 project evidence range','IRENA (2025), Renewable Power Generation Costs in 2024'],
    ['Geothermal','Total Installed Cost','cost',4015,1217,6724,'USD/kW','reported 2024 project evidence range','IRENA (2025), Renewable Power Generation Costs in 2024'],
    ['Geothermal','Capacity Factor','benefit',88,50,96,'%','reported 2010-2024 project evidence range','IRENA (2025), Renewable Power Generation Costs in 2024'],
    ['Geothermal','Life-cycle GHG','cost',37,22,52,'gCO2e/kWh','Q1-median-Q3; geothermal/all technologies','NREL public LCA Harmonization dataset (2021; DOI 10.7799/1819907)'],

    ['Bioenergy','LCOE','cost',0.087,0.065,0.106,'USD/kWh','reported 2000-2024 country/region evidence range','IRENA (2025), Renewable Power Generation Costs in 2024'],
    ['Bioenergy','Total Installed Cost','cost',3242,746,5864,'USD/kW','5th-95th percentile, 2000-2024 China all-feedstock evidence','IRENA (2025), Renewable Power Generation Costs in 2024'],
    ['Bioenergy','Capacity Factor','benefit',73,44,91,'%','5th-95th percentile, 2000-2024 regional evidence','IRENA (2025), Renewable Power Generation Costs in 2024'],
    ['Bioenergy','Life-cycle GHG','cost',52,28,110,'gCO2e/kWh','Q1-median-Q3; biopower/all technologies','NREL public LCA Harmonization dataset (2021; DOI 10.7799/1819907)'],
]
COLUMNS = ['alternative','criterion','direction','central','lower','upper','unit','range_semantics','source']

SOURCE_URLS = {
    'IRENA_2024_COSTS':'https://www.irena.org/Digital-Report/Renewable-Power-Generation-Costs-in-2024',
    'IRENA_2024_COSTS_PDF':'https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2025/Jul/IRENA_TEC_RPGC_in_2024_2025.pdf',
    'IPCC_AR5_CH9':'https://www.ipcc.ch/site/assets/uploads/2018/03/Chapter-9-Renewable-Energy-in-the-Context-of-Sustainable-Development-1.pdf',
    'NREL_LCA_DATA':'https://data.nrel.gov/submissions/171',
    'NREL_LCA_FACTSHEET':'https://www.nrel.gov/docs/fy21osti/80580.pdf',
    'ATANASSOV_1986':'https://doi.org/10.1016/S0165-0114(86)80034-3',
    'IF_TOPSIS_EXAMPLE':'https://www.mdpi.com/1099-4300/23/5/563',
}



SOURCE_REGISTRY = pd.DataFrame([
    ['IRENA_2024_COSTS','International Renewable Energy Agency (IRENA)','Renewable Power Generation Costs in 2024','2025','https://www.irena.org/Digital-Report/Renewable-Power-Generation-Costs-in-2024'],
    ['NREL_LCA_DATA','Nicholson, S.; Heath, G.','Life Cycle Emissions Factors for Electricity Generation Technologies','2021; public dataset last updated 2026','https://data.nrel.gov/submissions/171'],
    ['NREL_LCA_FACTSHEET','National Renewable Energy Laboratory','Life Cycle Greenhouse Gas Emissions from Electricity Generation: Update','2021','https://www.nrel.gov/docs/fy21osti/80580.pdf'],
    ['ATANASSOV_1986','Atanassov, K. T.','Intuitionistic fuzzy sets','1986','https://doi.org/10.1016/S0165-0114(86)80034-3'],
    ['IF_TOPSIS_EXAMPLE','Intuitionistic Fuzzy TOPSIS reference','Intuitionistic Fuzzy TOPSIS as a Method for Assessing Socioeconomic Phenomena on the Basis of Survey Data','2021','https://www.mdpi.com/1099-4300/23/5/563'],
    ['NREA','New and Renewable Energy Authority (Egypt)','Official reports and renewable-energy resource information','official public source','https://nrea.gov.eg/'],
])

def dataset_df():
    return pd.DataFrame(DATA_ROWS, columns=COLUMNS)


def validate_dataset(df):
    if list(df.columns) != COLUMNS:
        raise ValueError('Dataset schema mismatch.')
    if len(df) != len(ALTERNATIVES)*len(CRITERIA):
        raise ValueError('Expected 20 alternative-criterion observations.')
    if set(df.alternative) != set(ALTERNATIVES):
        raise ValueError('Alternative set mismatch.')
    if set(df.criterion) != set(CRITERIA):
        raise ValueError('Criterion set mismatch.')
    x = df[['central','lower','upper']].to_numpy(float)
    if not np.isfinite(x).all():
        raise ValueError('Non-finite numerical value found.')
    if np.any(df.lower.to_numpy(float) > df.upper.to_numpy(float)):
        raise ValueError('Lower bound exceeds upper bound.')
    if np.any(df.central.to_numpy(float) < df.lower.to_numpy(float)) or np.any(df.central.to_numpy(float) > df.upper.to_numpy(float)):
        raise ValueError('Central value is outside the published evidence range.')


def normalize_series(x, direction):
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    if np.isclose(lo, hi):
        return np.full_like(x, 0.5)
    if direction == 'benefit':
        return (x-lo)/(hi-lo)
    return (hi-x)/(hi-lo)


def central_matrix(df):
    out = pd.DataFrame(index=ALTERNATIVES, columns=CRITERIA, dtype=float)
    for c in CRITERIA:
        s = df[df.criterion == c].set_index('alternative').loc[ALTERNATIVES]
        out[c] = normalize_series(s.central.to_numpy(float), DIRECTIONS[c])
    return out


def interval_to_ifs(df, collapse=False):
    rows = []
    for c in CRITERIA:
        s = df[df.criterion == c].set_index('alternative').loc[ALTERNATIVES]
        lo = s.lower.to_numpy(float)
        hi = s.upper.to_numpy(float)
        direction = DIRECTIONS[c]
        # For each criterion, normalize the published evidence envelope
        # to a common [0,1] performance scale.
        gmin = lo.min()
        gmax = hi.max()
        if np.isclose(gmin, gmax):
            raise ValueError(f'Degenerate evidence envelope: {c}')
        if direction == 'benefit':
            l = (lo-gmin)/(gmax-gmin)
            u = (hi-gmin)/(gmax-gmin)
        else:
            l = (gmax-hi)/(gmax-gmin)
            u = (gmax-lo)/(gmax-gmin)
        if collapse:
            mid = (l+u)/2
            l, u = mid, mid
        for i,a in enumerate(ALTERNATIVES):
            mu = float(l[i]); nu = float(1-u[i]); pi = float(u[i]-l[i])
            if min(mu,nu,pi) < -1e-10 or abs(mu+nu+pi-1)>1e-9:
                raise ValueError(f'Invalid IFN for {a}/{c}: {(mu,nu,pi)}')
            rows.append([a,c,mu,nu,pi,pi,(1+mu-nu)/2])
    return pd.DataFrame(rows, columns=['alternative','criterion','mu','nu','pi','hesitation','score'])


def matrix_tensor(ifs_df):
    m, n = len(ALTERNATIVES), len(CRITERIA)
    D = np.zeros((m,n,3), dtype=float)
    for i,a in enumerate(ALTERNATIVES):
        for j,c in enumerate(CRITERIA):
            r = ifs_df[(ifs_df.alternative==a)&(ifs_df.criterion==c)].iloc[0]
            D[i,j] = [r.mu, r.nu, r.pi]
    return D


def entropy_weights(df):
    Z = central_matrix(df)
    m = len(ALTERNATIVES)
    P = pd.DataFrame(index=ALTERNATIVES, columns=CRITERIA, dtype=float)
    E = {}
    G = {}
    for c in CRITERIA:
        v = Z[c].to_numpy(float)
        s = v.sum()
        p = np.full(m,1/m) if s <= 1e-15 else v/s
        P[c] = p
        mask = p > 0
        terms = np.zeros_like(p)
        terms[mask] = p[mask] * np.log(p[mask])
        e = -terms.sum()/math.log(m)
        E[c] = float(e); G[c] = float(1-e)
    g = np.array([G[c] for c in CRITERIA], dtype=float)
    if g.sum() <= 1e-15: w = np.full(len(CRITERIA),1/len(CRITERIA))
    else: w = g/g.sum()
    return Z, P, pd.Series(E), pd.Series(G), pd.Series(w,index=CRITERIA)


def weighted_ifs_topsis(D, weights):
    w = np.asarray(weights,dtype=float)
    if len(w)!=D.shape[1] or not np.isclose(w.sum(),1):
        raise ValueError('Weights must match criteria and sum to 1.')
    # Data-driven ideal points using the IFS score ordering.
    S = (1+D[:,:,0]-D[:,:,1])/2
    pos = np.zeros((D.shape[1],3)); neg = np.zeros((D.shape[1],3))
    for j,c in enumerate(CRITERIA):
        if DIRECTIONS[c]=='benefit':
            ip, im = np.argmax(S[:,j]), np.argmin(S[:,j])
        else:
            ip, im = np.argmax(S[:,j]), np.argmin(S[:,j])
        pos[j] = D[ip,j]; neg[j] = D[im,j]
    dpos = np.zeros(D.shape[0]); dneg = np.zeros(D.shape[0])
    for i in range(D.shape[0]):
        sqp=0.0; sqn=0.0
        for j in range(D.shape[1]):
            djp = np.linalg.norm(D[i,j]-pos[j])/math.sqrt(2)
            djn = np.linalg.norm(D[i,j]-neg[j])/math.sqrt(2)
            sqp += w[j]*djp*djp
            sqn += w[j]*djn*djn
        dpos[i]=math.sqrt(max(sqp,0)); dneg[i]=math.sqrt(max(sqn,0))
    cc=dneg/(dpos+dneg+1e-15)
    return dpos,dneg,cc,pos,neg


def classical_topsis(Z, weights):
    X=Z.to_numpy(float); w=np.asarray(weights,float)
    V=X*w[None,:]
    pos=np.zeros(X.shape[1]); neg=np.zeros(X.shape[1])
    for j,c in enumerate(CRITERIA):
        if DIRECTIONS[c]=='benefit':
            pos[j]=V[:,j].max(); neg[j]=V[:,j].min()
        else:
            pos[j]=V[:,j].max(); neg[j]=V[:,j].min()
    dp=np.linalg.norm(V-pos,axis=1); dn=np.linalg.norm(V-neg,axis=1)
    cc=dn/(dp+dn+1e-15)
    return dp,dn,cc


def adjusted_weights(base, ifs_df, alpha=1.0):
    H=ifs_df.groupby('criterion')['hesitation'].mean().reindex(CRITERIA)
    R=np.clip(1-alpha*H.to_numpy(float),0,1)
    b=base.to_numpy(float)
    raw=b*R
    if raw.sum()<=1e-15: raise ValueError('All adjusted weights are zero.')
    w=raw/raw.sum()
    return H,pd.Series(R,index=CRITERIA),pd.Series(w,index=CRITERIA)


def kendall_tau(a,b):
    a=list(a); b=list(b)
    rb={v:i for i,v in enumerate(b)}
    ranks=[rb[v] for v in a]
    conc=disc=0
    for i in range(len(ranks)):
        for j in range(i+1,len(ranks)):
            if ranks[i]<ranks[j]: conc+=1
            elif ranks[i]>ranks[j]: disc+=1
    den=conc+disc
    return 1.0 if den==0 else (conc-disc)/den


def rank_series(values):
    order=np.argsort(-np.asarray(values))
    return [ALTERNATIVES[i] for i in order]



def central_matrix_vector(df):
    out=pd.DataFrame(index=ALTERNATIVES,columns=CRITERIA,dtype=float)
    for c in CRITERIA:
        s=df[df.criterion==c].set_index('alternative').loc[ALTERNATIVES]
        x=s.central.to_numpy(float)
        den=np.linalg.norm(x)
        if den<=1e-15:
            raise ValueError(f'Cannot vector-normalize {c}.')
        r=x/den
        out[c]=r if DIRECTIONS[c]=='benefit' else 1-r
    return out


def entropy_weights_from_Z(Z):
    m=len(ALTERNATIVES)
    P=pd.DataFrame(index=ALTERNATIVES,columns=CRITERIA,dtype=float)
    E={}; G={}
    for c in CRITERIA:
        v=Z[c].to_numpy(float); s=v.sum()
        p=np.full(m,1/m) if s<=1e-15 else v/s
        P[c]=p
        mask=p>0; terms=np.zeros_like(p); terms[mask]=p[mask]*np.log(p[mask])
        e=-terms.sum()/math.log(m)
        E[c]=float(e); G[c]=float(1-e)
    g=np.array([G[c] for c in CRITERIA],dtype=float)
    w=g/g.sum() if g.sum()>1e-15 else np.full(len(CRITERIA),1/len(CRITERIA))
    return P,pd.Series(E),pd.Series(G),pd.Series(w,index=CRITERIA)


def interval_to_ifs_vector(df, collapse=False):
    rows=[]
    for c in CRITERIA:
        s=df[df.criterion==c].set_index('alternative').loc[ALTERNATIVES]
        lo=s.lower.to_numpy(float); hi=s.upper.to_numpy(float)
        den=np.linalg.norm(s.central.to_numpy(float))
        if den<=1e-15:
            raise ValueError(f'Cannot vector-normalize interval for {c}.')
        if DIRECTIONS[c]=='benefit':
            l=lo/den; u=hi/den
        else:
            l=1-hi/den; u=1-lo/den
        l=np.clip(l,0,1); u=np.clip(u,0,1)
        if collapse:
            mid=(l+u)/2; l,u=mid,mid
        for i,a in enumerate(ALTERNATIVES):
            mu=float(l[i]); nu=float(1-u[i]); pi=float(u[i]-l[i])
            if min(mu,nu,pi)<-1e-10 or abs(mu+nu+pi-1)>1e-9:
                raise ValueError(f'Invalid vector-normalized IFN for {a}/{c}.')
            rows.append([a,c,mu,nu,pi,pi,(1+mu-nu)/2])
    return pd.DataFrame(rows,columns=['alternative','criterion','mu','nu','pi','hesitation','score'])


def normalization_sensitivity(df, baseline_cc):
    Z2=central_matrix_vector(df)
    _,_,_,w2=entropy_weights_from_Z(Z2)
    ifs2=interval_to_ifs_vector(df,collapse=False)
    D2=matrix_tensor(ifs2)
    _,_,cc_std2,_,_=weighted_ifs_topsis(D2,w2)
    H2,R2,w2adj=adjusted_weights(w2,ifs2,alpha=1.0)
    _,_,cc_ra2,_,_=weighted_ifs_topsis(D2,w2adj)
    return pd.DataFrame({
        'alternative':ALTERNATIVES,
        'minmax_RA_IF':baseline_cc,
        'vector_RA_IF':cc_ra2,
    }).assign(
        minmax_rank=lambda x:x['minmax_RA_IF'].rank(ascending=False,method='min').astype(int),
        vector_rank=lambda x:x['vector_RA_IF'].rank(ascending=False,method='min').astype(int),
    ), w2, w2adj, H2

def run_all():
    df=dataset_df(); validate_dataset(df)
    df.to_csv(RESULTS/'embedded_dataset.csv',index=False)
    audit=df[['alternative','criterion','central','lower','upper','unit','range_semantics','source']].copy()
    audit.to_csv(RESULTS/'dataset_audit.csv',index=False)
    SOURCE_REGISTRY.to_csv(RESULTS/'source_registry.csv',index=False)

    Z,P,E,G,w=entropy_weights(df)
    Z.to_csv(RESULTS/'normalized_central_matrix.csv')
    pd.DataFrame({'entropy_E':E,'divergence_1_minus_E':G,'base_weight':w}).to_csv(RESULTS/'criterion_weights_base.csv')

    ifs=interval_to_ifs(df,collapse=False)
    ifs0=interval_to_ifs(df,collapse=True)
    ifs.to_csv(RESULTS/'ifs_decision_matrix.csv',index=False)
    ifs0.to_csv(RESULTS/'ifs_ablation_matrix.csv',index=False)
    D=matrix_tensor(ifs); D0=matrix_tensor(ifs0)

    _,_,cc_classical=classical_topsis(Z,w)
    _,_,cc_if,_,_=weighted_ifs_topsis(D,w)
    _,_,cc_if0,_,_=weighted_ifs_topsis(D0,w)
    H,R,w_adj=adjusted_weights(w,ifs,alpha=1.0)
    _,_,cc_ra,_,_=weighted_ifs_topsis(D,w_adj)

    # Mathematical validity checks
    max_ifs_residual=float(np.max(np.abs(D.sum(axis=2)-1.0)))
    if max_ifs_residual>1e-9: raise AssertionError('IFS feasibility check failed.')
    if not np.isclose(w.sum(),1.0,atol=1e-10): raise AssertionError('Base weights do not sum to one.')
    if not np.isclose(w_adj.sum(),1.0,atol=1e-10): raise AssertionError('Adjusted weights do not sum to one.')
    if np.any((R.to_numpy()< -1e-12) | (R.to_numpy()>1+1e-12)): raise AssertionError('Reliability bounds violated.')

    validation = pd.DataFrame([
        ['max_IF_feasibility_residual', max_ifs_residual, max_ifs_residual <= 1e-9],
        ['base_weight_sum', float(w.sum()), bool(np.isclose(w.sum(),1.0,atol=1e-10))],
        ['adjusted_weight_sum', float(w_adj.sum()), bool(np.isclose(w_adj.sum(),1.0,atol=1e-10))],
        ['reliability_bounds', bool(np.all((R.to_numpy()>=-1e-12)&(R.to_numpy()<=1+1e-12))), True],
    ], columns=['check','value','passed'])
    validation.to_csv(RESULTS/'validation_checks.csv', index=False)

    ranking=pd.DataFrame({'alternative':ALTERNATIVES,
                          'Classical_TOPSIS':cc_classical,
                          'Degenerate_IF_TOPSIS':cc_if0,
                          'Standard_IF_TOPSIS':cc_if,
                          'Proposed_RA_IF_TOPSIS':cc_ra})
    for col in ranking.columns[1:]: ranking[col+'_rank']=ranking[col].rank(ascending=False,method='min').astype(int)
    ranking.to_csv(RESULTS/'ranking_comparison.csv',index=False)

    cw=pd.DataFrame({'base_weight':w,'mean_hesitation_H':H,'reliability_R':R,'adjusted_weight':w_adj})
    cw.to_csv(RESULTS/'criterion_weights_hesitation_reliability.csv')

    # Ablation: isolate the effect of pi.
    abl=ranking[['alternative','Classical_TOPSIS','Degenerate_IF_TOPSIS','Standard_IF_TOPSIS','Proposed_RA_IF_TOPSIS']].copy()
    abl.to_csv(RESULTS/'ifs_ablation.csv',index=False)

    # Hesitation sensitivity: scale the evidence-induced hesitation while preserving IFS feasibility.
    sens_rows=[]
    for alpha in [0,0.25,0.5,0.75,1.0]:
        Hx,Rx,wx=adjusted_weights(w,ifs,alpha=alpha)
        _,_,cc,_,_=weighted_ifs_topsis(D,wx)
        order=rank_series(cc)
        sens_rows.append({'alpha':alpha,'top_alternative':order[0],'top_score':float(cc[np.argmax(cc)]),
                          'rank_order':' > '.join(order),**{f'w_{c}':float(wx[c]) for c in CRITERIA}})
    sens=pd.DataFrame(sens_rows)
    sens.to_csv(RESULTS/'hesitation_sensitivity.csv',index=False)

    # Semantic sensitivity for GHG: baseline NREL Q1-Q3 vs NREL min-max.
    # Both scenarios come from the same public NREL LCA Harmonization dataset,
    # so the sensitivity isolates range semantics without changing the source.
    df_mm=df.copy()
    ghg_raw={
        'Solar PV':(20,217),'Onshore Wind':(1.3,81),'Hydropower':(0.57,75),'Geothermal':(5.6,245),'Bioenergy':(-1000,1300)
    }
    for a,(lo,hi) in ghg_raw.items():
        m=(df_mm.alternative==a)&(df_mm.criterion=='Life-cycle GHG')
        df_mm.loc[m,'lower']=lo; df_mm.loc[m,'upper']=hi
        df_mm.loc[m,'range_semantics']='NREL min-max sensitivity envelope'
    ifs_mm=interval_to_ifs(df_mm,collapse=False); Dmm=matrix_tensor(ifs_mm)
    Hm,Rm,wm=adjusted_weights(w,ifs_mm,alpha=1.0)
    _,_,cc_mm,_,_=weighted_ifs_topsis(Dmm,wm)
    sem=pd.DataFrame({'alternative':ALTERNATIVES,'baseline_IQR_RA_IF':cc_ra,'minmax_GHG_RA_IF':cc_mm})
    sem['baseline_rank']=sem.baseline_IQR_RA_IF.rank(ascending=False,method='min').astype(int)
    sem['minmax_rank']=sem.minmax_GHG_RA_IF.rank(ascending=False,method='min').astype(int)
    sem.to_csv(RESULTS/'ghg_semantic_sensitivity.csv',index=False)

    # Normalization sensitivity: baseline direction-aware min-max vs
    # direction-aware vector normalization. This is an auxiliary robustness test.
    norm_sens,w_vec,w_vec_adj,H_vec=normalization_sensitivity(df,cc_ra)
    norm_sens.to_csv(RESULTS/'normalization_sensitivity.csv',index=False)
    pd.DataFrame({'vector_base_weight':w_vec,'vector_adjusted_weight':w_vec_adj,'vector_mean_hesitation':H_vec}).to_csv(RESULTS/'normalization_vector_weights.csv')

    # Criterion-weight perturbation robustness: +/-10% one criterion at a time.
    wr=[]
    base_order=rank_series(cc_ra)
    for c in CRITERIA:
        j=CRITERIA.index(c)
        for factor in [0.90,1.00,1.10]:
            wp=w_adj.to_numpy(float).copy()
            wp[j]*=factor
            wp=wp/wp.sum()
            _,_,ccp,_,_=weighted_ifs_topsis(D,wp)
            order=rank_series(ccp)
            wr.append({'perturbed_criterion':c,'factor':factor,'top_alternative':order[0],
                       'rank_order':' > '.join(order),'kendall_tau_to_baseline':kendall_tau(base_order,order)})
    pd.DataFrame(wr).to_csv(RESULTS/'criterion_weight_robustness.csv',index=False)

    # Figures
    plt.rcParams.update({'figure.dpi':150})
    rplot=ranking.set_index('alternative')[['Classical_TOPSIS','Standard_IF_TOPSIS','Proposed_RA_IF_TOPSIS']]
    ax=rplot.plot(kind='bar',figsize=(10,5)); ax.set_ylabel('Closeness coefficient'); ax.set_title('TOPSIS comparison'); ax.tick_params(axis='x',rotation=35); ax.grid(axis='y',alpha=.25); ax.figure.tight_layout(); ax.figure.savefig(RESULTS/'01_ranking_comparison.png'); plt.close(ax.figure)

    ax=cw[['base_weight','adjusted_weight']].plot(kind='bar',figsize=(9,5)); ax.set_ylabel('Criterion weight'); ax.set_title('Criterion-weight adjustment'); ax.tick_params(axis='x',rotation=30); ax.grid(axis='y',alpha=.25); ax.figure.tight_layout(); ax.figure.savefig(RESULTS/'02_weight_adjustment.png'); plt.close(ax.figure)

    ax=cw['mean_hesitation_H'].plot(kind='bar',figsize=(9,5)); ax.set_ylabel('Mean hesitation H_j'); ax.set_title('Evidence-induced hesitation by criterion'); ax.tick_params(axis='x',rotation=30); ax.grid(axis='y',alpha=.25); ax.figure.tight_layout(); ax.figure.savefig(RESULTS/'03_hesitation_by_criterion.png'); plt.close(ax.figure)

    ax=abl.set_index('alternative')[['Degenerate_IF_TOPSIS','Standard_IF_TOPSIS']].plot(kind='bar',figsize=(10,5)); ax.set_ylabel('Closeness coefficient'); ax.set_title('IFS ablation: without vs with evidence-induced hesitation'); ax.tick_params(axis='x',rotation=35); ax.grid(axis='y',alpha=.25); ax.figure.tight_layout(); ax.figure.savefig(RESULTS/'04_ifs_ablation.png'); plt.close(ax.figure)

    ax=sens.set_index('alpha')[['w_LCOE','w_Total Installed Cost','w_Capacity Factor','w_Life-cycle GHG']].plot(marker='o',figsize=(10,5)); ax.set_ylabel('Adjusted criterion weight'); ax.set_title('Sensitivity to hesitation attenuation factor'); ax.grid(alpha=.25); ax.figure.tight_layout(); ax.figure.savefig(RESULTS/'05_hesitation_sensitivity.png'); plt.close(ax.figure)

    ax=sem.set_index('alternative')[['baseline_IQR_RA_IF','minmax_GHG_RA_IF']].plot(kind='bar',figsize=(10,5)); ax.set_ylabel('RA-IF-TOPSIS closeness'); ax.set_title('Robustness to GHG range semantics'); ax.tick_params(axis='x',rotation=35); ax.grid(axis='y',alpha=.25); ax.figure.tight_layout(); ax.figure.savefig(RESULTS/'06_ghg_semantic_sensitivity.png'); plt.close(ax.figure)

    ax=norm_sens.set_index('alternative')[['minmax_RA_IF','vector_RA_IF']].plot(kind='bar',figsize=(10,5)); ax.set_ylabel('RA-IF-TOPSIS closeness'); ax.set_title('Sensitivity to normalization method'); ax.tick_params(axis='x',rotation=35); ax.grid(axis='y',alpha=.25); ax.figure.tight_layout(); ax.figure.savefig(RESULTS/'07_normalization_sensitivity.png'); plt.close(ax.figure)

    # Statistical-free rank agreement table.
    orders={k:rank_series(ranking[k].to_numpy()) for k in ['Classical_TOPSIS','Degenerate_IF_TOPSIS','Standard_IF_TOPSIS','Proposed_RA_IF_TOPSIS']}
    pair=[]
    keys=list(orders)
    for i in range(len(keys)):
        for j in range(i+1,len(keys)):
            pair.append({'method_1':keys[i],'method_2':keys[j],'kendall_tau':kendall_tau(orders[keys[i]],orders[keys[j]])})
    pd.DataFrame(pair).to_csv(RESULTS/'ranking_agreement.csv',index=False)

    summary_lines=[]
    summary_lines.append('RA-IFS-TOPSIS FINAL RESEARCH PIPELINE: SUCCESS')
    summary_lines.append('Reference year: 2024; baseline GHG semantics: NREL Q1-median-Q3 quartile evidence range.')
    summary_lines.append('All computations completed from this single main.py file.')
    summary_lines.append('Top proposed alternative: '+rank_series(cc_ra)[0])
    summary_lines.append('Proposed ranking: '+' > '.join(rank_series(cc_ra)))
    summary_lines.append('Base weights: '+str(w.to_dict()))
    summary_lines.append('Adjusted weights: '+str(w_adj.to_dict()))
    summary_lines.append('Max IFS normalization residual: '+f'{max_ifs_residual:.3e}')
    summary_lines.append('Weight-sum checks: base='+f'{w.sum():.12f}'+', adjusted='+f'{w_adj.sum():.12f}')
    summary_lines.append('Mean hesitation: '+str(H.to_dict()))
    summary_lines.append('Semantic sensitivity top (baseline IQR): '+str(rank_series(cc_ra)[0]))
    summary_lines.append('Semantic sensitivity top (GHG min-max): '+str(rank_series(cc_mm)[0]))
    summary_lines.append('Normalization sensitivity top (vector): '+str(rank_series(norm_sens['vector_RA_IF'].to_numpy())[0]))
    (RESULTS/'run_summary.txt').write_text('\n'.join(summary_lines),encoding='utf-8')
    import hashlib, json, zipfile
    manifest={}
    for f in sorted(RESULTS.iterdir()):
        if f.is_file():
            manifest[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
    (RESULTS/'reproducibility_manifest.json').write_text(json.dumps({
        'main_script':Path(__file__).name,
        'generated_files':manifest,
    },indent=2),encoding='utf-8')
    bundle=RESULTS/'results_bundle.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(RESULTS.iterdir()):
            if f.is_file() and f.name != bundle.name:
                zf.write(f,arcname=f.name)
    print('='*80); print('RA-IFS-TOPSIS FINAL PIPELINE: SUCCESS'); print('='*80)
    print('\nRanking comparison:'); print(ranking.to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    print('\nWeights / hesitation / reliability:'); print(cw.to_string(float_format=lambda x:f'{x:.6f}'))
    print('\nProposed ranking:', ' > '.join(rank_series(cc_ra)))
    print('\nGHG semantic sensitivity:'); print(sem.to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    print('\nNormalization sensitivity:'); print(norm_sens.to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    return ranking,cw,ifs,sens,sem

# ---- helper versions for leave-one-out ----
def matrix_tensor_custom(ifs_df, alternatives):
    D=np.zeros((len(alternatives),len(CRITERIA),3))
    for i,a in enumerate(alternatives):
        for j,c in enumerate(CRITERIA):
            r=ifs_df[(ifs_df.alternative==a)&(ifs_df.criterion==c)].iloc[0]; D[i,j]=[r.mu,r.nu,r.pi]
    return D

def adjusted_weights_custom(base,ifs_df,alternatives,alpha=1.0):
    H=ifs_df.groupby('criterion').hesitation.mean().reindex(CRITERIA)
    R=np.clip(1-alpha*H.to_numpy(float),0,1)
    raw=base.to_numpy(float)*R
    return H,pd.Series(R,index=CRITERIA),pd.Series(raw/raw.sum(),index=CRITERIA)

def weighted_ifs_topsis_custom(D,weights,alternatives):
    w=np.asarray(weights,float); S=(1+D[:,:,0]-D[:,:,1])/2
    pos=np.zeros((len(CRITERIA),3)); neg=np.zeros((len(CRITERIA),3))
    for j,c in enumerate(CRITERIA):
        ip,im=np.argmax(S[:,j]),np.argmin(S[:,j]); pos[j]=D[ip,j]; neg[j]=D[im,j]
    dp=[]; dn=[]
    for i in range(D.shape[0]):
        sp=sn=0
        for j in range(D.shape[1]):
            a=np.linalg.norm(D[i,j]-pos[j])/math.sqrt(2); b=np.linalg.norm(D[i,j]-neg[j])/math.sqrt(2)
            sp+=w[j]*a*a; sn+=w[j]*b*b
        dp.append(math.sqrt(sp)); dn.append(math.sqrt(sn))
    dp=np.array(dp); dn=np.array(dn); return dp,dn,dn/(dp+dn+1e-15),pos,neg

if __name__ == '__main__':
    run_all()
