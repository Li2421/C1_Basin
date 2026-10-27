"""Diagnose saved route-unknown states without stepping or changing the plant."""
from core import *
from scenes import geometry
from routes import route_geometry
from single_integrator.environment import point_segment_distance

def main():
    frozen=read(HERE/'SCENES_FROZEN.json')
    scenes={e['scene']['name']:e['scene']for e in frozen['entries']if 'scene'in e}
    output=[];geometry_cache={}
    for p in sorted((HERE/'raw').glob('*.json')):
        r=read(p)
        if r['scene']not in scenes or not r['numerical_failure']or r.get('safety_layer')!='distributed':continue
        if 'route'not in (r.get('error')or{}).get('message','').lower():continue
        s=scenes[r['scene']]
        if r['scene']not in geometry_cache:geometry_cache[r['scene']]=route_geometry(geometry(s))
        geom=geometry_cache[r['scene']]
        with np.load(r['trace'])as a:
            positions=a['positions'][-1];goals=a['goals'];points=[]
            certificates=json.loads(str(a['safety_certificates_json']))if 'safety_certificates_json'in a else []
            for i,(position,goal)in enumerate(zip(positions,goals)):
                valid=geom.valid_points(np.asarray([position,goal]))
                if valid.all()and geom.path(position,goal)is not None:continue
                distances=point_segment_distance(position[None],geom.walls[:,0],geom.walls[:,1])
                clearance=float(np.min(distances))-float(geom.margin)
                geometric_witness=None
                # Diagnostic only: a visible short connector to a point where
                # the original roadmap succeeds. Never supplied to any controller.
                if valid.all():
                    for radius in(.02,.05,.1,.2,.4):
                        for angle in np.arange(32)*2*np.pi/32:
                            q=position+radius*np.array([np.cos(angle),np.sin(angle)])
                            if not geom.valid_points(q[None])[0]or not geom.visible(position,q):continue
                            tail=geom.path(q,goal)
                            if tail is not None:
                                geometric_witness=np.vstack((position,tail));break
                        if geometric_witness is not None:break
                points.append(dict(agent=i,position=position,goal=goal,start_valid=bool(valid[0]),goal_valid=bool(valid[1]),minimum_wall_distance=float(np.min(distances)),route_margin=float(geom.margin),clearance_above_route_margin=clearance,nearest_wall_index=int(np.argmin(distances)),path_none=geom.path(position,goal)is None,original_roadmap_visible_nodes=sum(geom.visible(position,q)for q in geom.nodes),diagnostic_short_connector_path=geometric_witness,connector_used_for_control=False))
            output.append(dict(uid=r['rollout_uid'],type_id=r['type_id'],controller=r['controller'],state_index=r['state_index'],purpose=r['purpose'],solo=r['solo'],step=(r.get('error')or{}).get('step'),spec=r['spec'],points=points,last_projection=certificates[-1]if certificates else None,reported_constraint_violation_steps=r.get('constraint_violation_steps'),reported_restoration_steps=r.get('restoration_steps'),reported_minimum_swept_clearance=r.get('minimum_swept_clearance')))
    dump(HERE/'ROUTE_UNKNOWN_DIAGNOSTICS.json',dict(cases=output,n=len(output),scope='Read-only original Geometry.path/valid_points applied to final saved positions; no rollout, solver change, geometry relaxation or retry. A negative route-margin clearance is an interface/applicability unknown, not automatically a physical collision or deadlock.'))
    print('route-unknown saved-state audit',len(output),flush=True)
    return output

if __name__=='__main__':main()
