"""Repair a metric-map grid-domain contract without changing paper control laws."""
from core import *

def main():
    source=HERE/'ports/impc_adapter.py';before=source.read_text();after=before
    after=after.replace('ys=lo[1]+np.arange(w)*SET.resolution','ys=lo[1]+np.arange(w+1)*SET.resolution')
    after=after.replace('corners=occupied[:-1]|occupied[1:];grid=corners.copy();grid[:,:-1]|=corners[:,1:];SET.path_obstacle_list=grid.astype(int)',
                        'corners=occupied[:-1]|occupied[1:];grid=corners[:,:-1]|corners[:,1:];SET.path_obstacle_list=grid.astype(int)')
    anchor='        import path_planning\n'
    guard='''        import path_planning
        original_validity=path_planning.ValidityChecker.isValid
        def grid_domain_validity(checker,state):
            # Original continuous map bounds can include a <resolution strip
            # beyond floor-sized occupancy storage. That strip is unavailable,
            # never a new free cell. Preserve the original in-range checker.
            x=int((state[0]-SET.map_range['x'][0])/SET.resolution)
            y=int((state[1]-SET.map_range['y'][0])/SET.resolution)
            if not (0<=x<SET.path_obstacle_list.shape[0] and 0<=y<SET.path_obstacle_list.shape[1]):return False
            return original_validity(checker,state)
        path_planning.ValidityChecker.isValid=grid_domain_validity
'''
    assert after.count(anchor)==1;after=after.replace(anchor,guard)
    target=HERE/'ports/impc_adapter_v2.py'
    if target.exists():assert target.read_text()==after
    else:target.write_text(after)
    dump(HERE/'IMPC_ADAPTER_V2.json',dict(source=str(source),original_sha256=sha(source),target=str(target),target_sha256=sha(target),changes=['Restore all four corner rows of the original geometry.grid_width occupancy rule.','Reject query indices outside the original allocated occupancy grid, including non-grid-aligned upper-bound slivers.'],unchanged=['Author ABITstar planner','Dynamic priority','Right-hand penalty','Safe-corridor construction and constraints','Physical walls/radius/dynamics/success','Frozen parameter options'],reason='Original author ValidityChecker checks continuous bounds but then indexes a floor-sized grid. New metric map extents need the missing grid-domain guard. Old index exceptions remain archived under their original exact controller IDs.'))

if __name__=='__main__':main()
