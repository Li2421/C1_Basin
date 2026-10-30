"""Pre-data selection and paired statistics; no simulator or model imports."""
import guard
from dataclasses import dataclass
from fractions import Fraction
import collections, itertools, math, random

BOOTSTRAP_SEED=2026102905
SIGNFLIP_SEED=2026102906

@dataclass(frozen=True)
class Score:
    index:int
    success:Fraction
    collisions:int
    unhandled:int
    truncated:float
    scenes:int
    episodes:int
    def key(self):return(-self.success,self.collisions,self.unhandled,self.truncated,self.index)
    def json(self):return dict(index=self.index,scene_equal_success=float(self.success),success_fraction=[self.success.numerator,self.success.denominator],collisions=self.collisions,unhandled=self.unhandled,scene_equal_truncated=self.truncated,scenes=self.scenes,episodes=self.episodes)

def score(index,rows,expected_keys=None):
    """Rows identify scene/state/nominal; every scheduled episode must be present."""
    keys=[(r['scene'],r['state'],r.get('nominal','paper'))for r in rows]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate selection evidence')
    if expected_keys is not None and set(keys)!=set(expected_keys):raise ValueError('Incomplete selection evidence; missing is not failure')
    if not rows:raise ValueError('No selection evidence')
    by_scene=collections.defaultdict(list)
    for row in rows:
        if row.get('result',row)is None:raise ValueError('Missing selection result')
        by_scene[row['scene']].append(row.get('result',row))
    success=[];truncated=[];collisions=unhandled=0
    for values in by_scene.values():
        assert all(r['success']in(0,1)for r in values)
        success.append(Fraction(sum(int(r['success'])for r in values),len(values)))
        times=[float(r['truncated_time'])for r in values]
        if not all(math.isfinite(x)for x in times):raise ValueError('Nonfinite selection time')
        truncated.append(math.fsum(times)/len(times))
        collisions+=sum(int(r['collision'])for r in values)
        unhandled+=sum(bool(r.get('paper_original',{}).get('numerical_unknown',r.get('numerical_failure',0)))for r in values)
    return Score(index,sum(success,Fraction())/len(success),collisions,unhandled,math.fsum(truncated)/len(truncated),len(by_scene),len(rows))

def first_round(scores):
    if len(scores)<4:raise ValueError('Fewer than four complete candidates')
    return sorted(scores,key=lambda s:s.key())[:4]

def final_choice(shortlist,second_round):
    """Cached outcomes for candidates outside the shortlist cannot enter selection."""
    ids=[s.index for s in shortlist]
    if not all(i in second_round for i in ids):raise ValueError('Incomplete second round')
    return min((second_round[i]for i in ids),key=lambda s:s.key())

def expected_max_curve(values):
    """Exact random-subset expectation, including ties; no Monte Carlo estimate."""
    ordered=sorted(x if isinstance(x,Fraction)else Fraction(str(x))for x in values);n=len(ordered)
    if not n:raise ValueError('Empty parameter menu')
    output=[]
    for k in range(1,n+1):
        denominator=math.comb(n,k)
        value=sum((ordered[i-1]*Fraction(math.comb(i-1,k-1),denominator)for i in range(k,n+1)),Fraction())
        output.append(dict(k=k,expected_max=float(value),fraction=[value.numerator,value.denominator]))
    return output

def saturation(values):
    if len(values)!=16:raise ValueError('Saturation decision requires the initial sixteen settings')
    curve=expected_max_curve(values);at8=Fraction(*curve[7]['fraction']);at16=Fraction(*curve[15]['fraction']);gain=at16-at8
    return dict(expand=gain>=Fraction(3,100),gain=float(gain),gain_fraction=[gain.numerator,gain.denominator],threshold_fraction=[3,100],curve=curve)

def collapse_nominals(rows,nominal_ids):
    """Return one value per physical initial state; do not resample controllers."""
    grouped=collections.defaultdict(dict);expected=set(nominal_ids)
    for row in rows:
        key=(row['scene'],row['state']);name=row['nominal']
        if name in grouped[key]:raise ValueError('Duplicate nominal result')
        value=row['value'];grouped[key][name]=value if isinstance(value,Fraction)else Fraction(value)
    output={}
    for key,values in grouped.items():
        if set(values)!=expected:raise ValueError('Incomplete nominal group; missing is not failure')
        output[key]=sum(values.values(),Fraction())/len(expected)
    return output

def holm(pvalues):
    """Keep all preregistered comparisons in the family, including missing tests."""
    order=sorted(pvalues,key=lambda k:(1. if pvalues[k]is None else pvalues[k],k));m=len(order);previous=0.;out={}
    for rank,key in enumerate(order):
        value=1. if pvalues[key]is None else float(pvalues[key])
        if not 0<=value<=1:raise ValueError('Invalid p value')
        previous=max(previous,min(1.,(m-rank)*value));out[key]=None if pvalues[key]is None else previous
    return out

def paired_signflip(scene_differences,seed=SIGNFLIP_SEED,draws=100000):
    """Two-sided scene-level sign test, with exact integer tie comparisons."""
    values=[x if isinstance(x,Fraction)else Fraction(x)for x in scene_differences]
    if not values:return dict(p=None,scenes=0)
    denominator=math.lcm(*(x.denominator for x in values));integers=[x.numerator*(denominator//x.denominator)for x in values];observed=abs(sum(integers));n=len(values);exact=n<=18
    count=2**n if exact else draws;rng=random.Random(seed);extreme=0
    for i in range(count):
        bits=i if exact else rng.getrandbits(n)
        statistic=abs(sum(value if bits&(1<<j)else-value for j,value in enumerate(integers)))
        extreme+=statistic>=observed
    return dict(p=extreme/count if exact else(extreme+1)/(count+1),scenes=n,exact=exact,draws=count,extreme=extreme,seed=seed,two_sided=True,comparison='absolute sum >= observed absolute sum, exact integer arithmetic')

def percentile(values,p):
    xs=sorted(values);where=(len(xs)-1)*p;left=math.floor(where);right=math.ceil(where)
    return xs[left]+(where-left)*(xs[right]-xs[left])

def hierarchical_interval(blocks,seed=BOOTSTRAP_SEED,resamples=10000):
    """category -> scene -> per-physical-state values (nominals already averaged)."""
    if not blocks:return dict(mean=None,ci95=None,physical_states=0,scenes=0)
    groups=[];values=[]
    for category in sorted(blocks):
        scenes=[list(map(float,blocks[category][s]))for s in sorted(blocks[category])]
        if any(not x for x in scenes):raise ValueError('Missing scene is not an empty/failed scene')
        if not scenes:raise ValueError('Empty category')
        groups.append(scenes)
        for scene in scenes:values.extend(scene)
    rng=random.Random(seed);samples=[]
    for _ in range(resamples):
        total=0.;n=0
        for scenes in groups:
            for _ in range(len(scenes)):
                selected=rng.choice(scenes)
                total+=math.fsum(rng.choice(selected)for _ in range(len(selected)));n+=len(selected)
        samples.append(total/n)
    return dict(mean=math.fsum(values)/len(values),ci95=[percentile(samples,.025),percentile(samples,.975)],physical_states=len(values),scenes=sum(map(len,groups)),resamples=resamples,seed=seed,unit='Scene resampling within category, then physical-state resampling; nominal controllers stay paired.')

def verdict(adjusted_p,ci95):
    if ci95 is None:return'缺失'
    if adjusted_p is not None and adjusted_p<.05 and ci95[0]>0:return'优于'
    if ci95[0]>-.05:return'不劣于'
    return'无法区分'
