"""Exact rational inequalities and rigorous scalar-volume enclosures; no simulation."""
from fractions import Fraction as Q
from math import isqrt,comb
from decimal import Decimal,localcontext,ROUND_FLOOR,ROUND_CEILING
from pathlib import Path
import json
P=Path(__file__).resolve().parent
def interval_json(lo,hi):
    def render(x,rounding):
        with localcontext() as ctx:
            ctx.prec=42;ctx.rounding=rounding
            return str(Decimal(x.numerator)/Decimal(x.denominator))
    return dict(lower_decimal=render(lo,ROUND_FLOOR),upper_decimal=render(hi,ROUND_CEILING),
                lower_rational=str(lo),upper_rational=str(hi))
def sqrt_enclose(x,places=45):
    scale=10**places;k=isqrt(x.numerator*scale*scale//x.denominator)
    lo=Q(k,scale);hi=Q(k+1,scale)
    assert lo*lo<=x<hi*hi
    return lo,hi
def atan_enclose(x,N=50):
    assert 0<x<1 and N%2==0
    lo=sum(((-1)**n*x**(2*n+1)/Q(2*n+1) for n in range(N)),Q(0))
    return lo,lo+x**(2*N+1)/Q(2*N+1)
def main():
    A=json.loads((P/'agent_a/geometry_contract.json').read_text())
    # Certify actual loaded binary64 parameter values as exact real inputs.
    R=Q.from_float(A['constants_m']['R_w']);Ds=Q.from_float(A['constants_m']['D_s'])
    w=Q.from_float(A['runtime_config']['corridor_width']/2)
    top=Q.from_float(A['runtime_config']['bay_top']);length=Q.from_float(A['runtime_config']['corridor_half_length'])
    h=w-R;H=top-R;a=length-R
    b=2*h;L=2*a;x=b/Ds;q=Ds*Ds-b*b
    sl,su=sqrt_enclose(q)
    N=24
    asinlo=sum((Q(comb(2*n,n),4**n*(2*n+1))*x**(2*n+1) for n in range(N)),Q(0))
    # All omitted asin coefficients are positive <=1.
    asinhi=asinlo+x**(2*N+1)/(1-x*x)
    poly=(L*L+Ds*Ds)*b*b/2-b**4/12
    coeff=b*b/2+q/3
    Jlo=coeff*sl+b*Ds*Ds*asinlo/2-Ds**3/3
    Jhi=coeff*su+b*Ds*Ds*asinhi/2-Ds**3/3
    vlo=poly-2*L*Jhi;vhi=poly-2*L*Jlo
    t5=atan_enclose(Q(1,5));t239=atan_enclose(Q(1,239))
    pilo=16*t5[0]-4*t239[1];pihi=16*t5[1]-4*t239[0]
    base=4*a*h+2*w*R+2*h*(H-w)
    area_lo=base-pihi*R*R/2;area_hi=base-pilo*R*R/2
    ve=Q(3,10)*Q(1,20)*Q(1,50)*Q(1,25)
    vp=Q(1,20)*Q(1,50)*Q(2,25)*Q(99,100)
    # B/C proof margins, represented as exact rationals.
    vY=Q(3,100);vP=Q(3,50);vfast=Q(1,10);t=Q(1,5)
    alpha=1-R/(2*t)
    margins=dict(yield_pair=Q(325,1000)**2-Ds**2-2*Q(2,100)*vY,
        yield_corner_allowed_squared_minus_vY_squared=Q(175,1000)**2-R**2-vY**2,
        yield_top=Q(68,100)-Q(4,10)-R-vY,
        pass_corner_quadratic_min=t*(t-R)-vP*vP/(4*alpha),
        faster_pass_corner_quadratic_min=t*(t-R)-vfast*vfast/(4*alpha),
        pass_pair=Q(38,100)**2-Ds**2-vP*vP,
        faster_pass_pair=Q(38,100)**2-Ds**2-vfast*vfast,
        approach_corner_squared_margin=Q(18,100)**2-R**2-Q(5,100)**2,
        approach_floor_velocity_margin=h-Q(26,1000),
        minimum_pass_wall=Q(175,1000)-R,
        minimum_entry_wall=Q(18,100)-R)
    assert all(v>0 for v in margins.values())
    assert 0<vlo<vhi and 0<area_lo<area_hi
    d=json.loads((P/'agent_d/results.json').read_text())
    assert abs(d['volumes']['conflict_domain_m4']-float((vlo+vhi)/2))<1e-15
    assert abs(d['volumes']['single_agent_area_m2']-float((area_lo+area_hi)/2))<1e-15
    # Directed conservative enclosures for the original eta* raw bounds in K_pass.
    g1=Q.from_float(A['goals'][0][0])-Q(365,1000)
    g2=-Q.from_float(A['goals'][1][0])-Q(25,1000)
    n1=sqrt_enclose(g1*g1+Q(2,100)**2)
    n2=sqrt_enclose(g2*g2+Q(46,100)**2)
    nr=sqrt_enclose(Q(675,1000)**2+Q(38,100)**2)
    gaplo=(g1/(2*n1[1])+g2/(2*n2[1]))/2-Q(675,1000)/(2*nr[0])
    gaphi=(g1/(2*n1[0])+g2/(2*n2[0]))/2-Q(675,1000)/(2*nr[1])
    ylo=(Q(38,100)/(2*nr[1])-Q(46,100)/(2*n2[0]))/2
    yhi=(Q(38,100)/(2*nr[0])-Q(46,100)/(2*n2[1]))/2
    assert gaplo>Q(4371005,100000000) and ylo>Q(2351227,100000000)
    B=json.loads((P/'agent_b/results.json').read_text());roots=[]
    for row in B['raw_seed_fixed_family']:
        z=Q.from_float(row['z_min_m']);eps=Q(5,10**12);nu=Q.from_float(row['raw_margin'])
        def raw_enclosure(zz):
            nn=sqrt_enclose(Q.from_float(1.09)**2+zz*zz)
            return zz/2*(1-Q(1,2)/nn[0]),zz/2*(1-Q(1,2)/nn[1])
        fl=raw_enclosure(z-eps);fu=raw_enclosure(z+eps)
        assert fl[1]<nu<fu[0]
        roots.append(dict(raw_margin=row['raw_margin'],minimal_z_enclosure_m=interval_json(z-eps,z+eps),
            certificate='monotone raw formula, upper f(left)<nu<lower f(right), all rational sqrt enclosures'))
    out=dict(status='NUMERICALLY CERTIFIED by exact Fraction arithmetic, bounded positive asin tail and alternating Machin pi series',
        constants=dict(R_w=str(R),h=str(h),H=str(H),a=str(a),D_s=str(Ds)),
        input_convention='Exact real interpretation of checkpoint-loaded binary64 wall endpoints and effective radii; ordinary controller arithmetic retains original floating tolerances',
        exact_volumes_m4=dict(K_entry=str(ve),K_pass=str(vp),ordinary_reference_box=str(Q(115,1000)**2*Q(4,100)**2)),
        conflict_domain_volume_m4=interval_json(vlo,vhi),single_agent_area_m2=interval_json(area_lo,area_hi),
        entry_fraction_of_conflict=interval_json(ve/vhi,ve/vlo),
        full_joint_volume_bounds_m4=interval_json(2*vlo,area_hi**2),
        entry_fraction_of_full_joint_bounds=interval_json(ve/area_hi**2,ve/(2*vlo)),
        pass_fraction_of_full_joint_bounds=interval_json(vp/area_hi**2,vp/(2*vlo)),
        exact_strict_margins={k:dict(rational=str(v),decimal=float(v)) for k,v in margins.items()},
        raw_pass_gap_conservative_bound_enclosure=interval_json(gaplo,gaphi),
        raw_pass_yield_conservative_bound_enclosure=interval_json(ylo,yhi),
        safe_rounded_raw_bounds_m_per_s={'pass_gap':.04371005,'yield':.02351227},
        raw_seed_root_enclosures=roots,
        necessity=dict(max_main_channel_z_m=float(2*h),minimum_upper_robot_height_at_order_crossing_m=float(Ds-h)),
        method='Vconf polynomial minus 2L times analytic circular integral; asin 24 terms with geometric tail; sqrt exact isqrt rational enclosure; pi=16atan(1/5)-4atan(1/239), 50 alternating terms',
        new_flow=0,new_projection=0,new_rollouts=0)
    (P/'arithmetic_certificate.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ('conflict_domain_volume_m4','entry_fraction_of_conflict','necessity')},indent=2))
if __name__=='__main__':main()
