"""CPU Pillow renderer; reads immutable full simulator traces only."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def box_faces(box):
    lo,hi=box
    v=np.array([[lo[0],lo[1],lo[2]],[hi[0],lo[1],lo[2]],[hi[0],hi[1],lo[2]],[lo[0],hi[1],lo[2]],
                [lo[0],lo[1],hi[2]],[hi[0],lo[1],hi[2]],[hi[0],hi[1],hi[2]],[lo[0],hi[1],hi[2]]])
    return [v[list(ids)] for ids in [(0,1,2,3),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]]


def render_pair(paths, folder, seed, pipeline_mode=False):
    folder.mkdir(parents=True,exist_ok=True)
    traces,records=[],[]
    for path in paths:
        with np.load(path,allow_pickle=False) as z:
            trace={k:z[k].copy() for k in z.files}
        record=json.loads(path.with_suffix(".json").read_text())
        assert sha(path)==record["trace_sha256"]
        assert len(trace["positions"])==record["episode_length"]+1
        assert np.array_equal(trace["steps"],np.arange(len(trace["positions"])))
        assert len(trace["swept_clearance"])==record["episode_length"]
        assert np.min(trace["swept_clearance"])>=-1e-9
        traces.append(trace); records.append(record)
    assert records[1]["outcome"]=="success"
    assert records[0]["seed"]==records[1]["seed"]==seed
    assert np.array_equal(traces[0]["positions"][0],traces[1]["positions"][0])
    assert np.array_equal(traces[0]["goals"],traces[1]["goals"])
    assert records[0]["initial_velocities"]==records[1]["initial_velocities"]
    assert records[0]["config"]==records[1]["config"]
    if not pipeline_mode:
        assert records[0]["outcome"]=="deadlock"
        assert records[0]["initial_state_sha256"]==records[1]["initial_state_sha256"]
        assert np.array_equal(traces[0]["positions"],traces[1]["positions"][:len(traces[0]["positions"])])
    font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font=lambda size:ImageFont.truetype(font_path,size)
    small,normal,title=font(14),font(18),font(24)
    colors=["#d55d30","#267bba"]; width,height=1200,650
    theta=np.linspace(0,2*np.pi,40); circles=[]
    for k in range(3):
        circle=np.zeros((40,3)); circle[:,(k+1)%3]=np.cos(theta)*records[0]["config"]["radius"]
        circle[:,(k+2)%3]=np.sin(theta)*records[0]["config"]["radius"]; circles.append(circle)

    def project(points,panel):
        p=np.asarray(points)
        return [tuple(v) for v in np.column_stack((panel*600+330+65*p[...,0]-75*p[...,1],
                                                   455+13*p[...,0]-35*p[...,1]-120*p[...,2]))]

    background=Image.new("RGB",(width,height),"#f5f7fa"); d=ImageDraw.Draw(background)
    d.text((25,18),"3D drone bottleneck | matched baseline and recovery",font=title,fill="#172432")
    subtitle="Flow + latched eta + explicit near-goal servo | no online route teacher" if pipeline_mode else "Raised side alcove: retreat, climb, move sideways, let the other drone pass"
    d.text((25,55),subtitle,font=normal,fill="#465769")
    d.line((600,105,600,515),fill="#cbd5df",width=2)
    for panel,trace in enumerate(traces):
        if pipeline_mode:
            baseline="Baseline: Flow + servo, eta = 0" if "placement" in records[0] else "Baseline: goal attraction + hard CBF"
            label=[baseline,"MACFlow + eta + hard CBF + goal servo"][panel]
        else:label=["Baseline: goal attraction + hard CBF","Intervention: explicit route witness"][panel]
        d.text((panel*600+22,105),label,font=normal,fill="#172432")
        if pipeline_mode:
            record=records[panel]
            cp=(record.get("flow_checkpoint_sha256") or "none")[:12]
            d.text((panel*600+22,204),f"checkpoint {cp} | eta {record.get('eta')} | Flow future {record.get('future_index','n/a')}",font=small,fill="#465769")
        layer=Image.new("RGBA",background.size,(0,0,0,0)); ld=ImageDraw.Draw(layer)
        for box in trace["obstacles"]:
            for face in box_faces(box):
                ld.polygon(project(face,panel),fill=(121,140,157,22),outline=(99,121,140,100))
        background=Image.alpha_composite(background.convert("RGBA"),layer).convert("RGB"); d=ImageDraw.Draw(background)
        xy=project(np.array([[-.85,.60,.87]]),panel)[0]
        d.text((xy[0]-30,xy[1]-35),"raised bay",font=small,fill="#526273")
        origin=np.array([[-2.65,-.3,0.]])
        for k,label in enumerate("xyz"):
            end=origin.copy(); end[0,k]+=.5
            a,b=project(np.concatenate([origin,end]),panel)
            d.line([a,b],fill="#536575",width=2); d.text(b,label,font=small,fill="#536575")
        for i,g in enumerate(trace["goals"]):
            x,y=project(g[None],panel)[0]
            d.ellipse((x-5,y-5,x+5,y+5),outline=colors[i],width=2)
            d.text((x-20,y+10),f"goal {i+1}",font=small,fill=colors[i])
    footer=["Deadlock: both unfinished; 5 s speed < 0.005 m/s, path < 0.02 m; close pair + opposing nominal velocities.",
            "Safety: continuous sphere/solid and pair-ellipsoid checks. Ideal velocity tracking; no flight dynamics.",
            "Every physical step retained, 10 fps at 1x speed. Baseline holds its terminal state. Display axes use unequal scales.",
            "Orange: drone 1. Blue: drone 2. Terminal servo activates within 0.6 m of a goal with a clear straight path." if pipeline_mode else "Orange: drone 1 yields into the raised side alcove. Blue: drone 2 passes through the shared corridor."]
    for k,line in enumerate(footer):d.text((25,533+k*26),line,font=small,fill="#465769")

    def frame_image(frame):
        im=background.copy(); d=ImageDraw.Draw(im)
        for panel,(trace,record) in enumerate(zip(traces,records)):
            t=min(frame,len(trace["positions"])-1)
            for i in range(2):
                xyz=trace["positions"][:t+1,i]
                if t:d.line(project(xyz,panel),fill=colors[i],width=3)
                for circle in circles:d.line(project(trace["positions"][t,i]+circle,panel),fill=colors[i],width=2)
            state=record["outcome"] if t==len(trace["positions"])-1 else "running"
            d.text((panel*600+22,145),f"seed {seed} | t = {t*record['config']['dt']:.1f} s | {state}",font=normal,fill="#172432")
            if t:
                margins=trace["swept_clearance"][:t].min(axis=0)
                d.text((panel*600+22,175),f"wall >= {margins[0]:.3f} m | pair ellipsoid gap >= {margins[2]:.4f}",font=small,fill="#465769")
        return im

    frames=max(len(t["positions"]) for t in traces)
    png=folder/f"seed{seed}_overview.png"; frame_image(frames-1).save(png)
    video=folder/f"seed{seed}_full_comparison.mp4"
    command=["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{width}x{height}",
             "-r","10","-i","-","-an","-c:v","libx264","-threads","1","-pix_fmt","yuv420p","-crf","23",str(video)]
    proc=subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
        for frame in range(frames):proc.stdin.write(frame_image(frame).tobytes())
    finally:
        proc.stdin.close()
    if proc.wait()!=0:raise RuntimeError("ffmpeg failed")
    probe=json.loads(subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0","-count_frames",
        "-show_entries","stream=nb_read_frames,r_frame_rate,duration","-of","json",str(video)]))
    assert int(probe["streams"][0]["nb_read_frames"])==frames
    manifest=dict(scope="single-scene development video; not the four-scene batch gate",seed=seed,
                  frames=frames,fps=10,complete=True,initial_state_sha256=records[0]["initial_state_sha256"],
                  sources=[dict(trace=str(p.resolve()),sha256=sha(p),record=str(p.with_suffix('.json').resolve())) for p in paths],
                  renderer_sha256=sha(Path(__file__)),video=str(video.resolve()),video_sha256=sha(video),
                  overview=str(png.resolve()),overview_sha256=sha(png),ffprobe=probe,
                  full_episodes=True,skip_frames=False,interpolated_positions=False,
                  controllers=[r["controller_uid"] for r in records],checkpoints=[r.get("flow_checkpoint_sha256") for r in records],
                  actor_variants=[r.get("actor_variant",r.get("controller")) for r in records],
                  etas=[r.get("eta") for r in records],future_indices=[r.get("future_index") for r in records],
                  terminal_events=[r["outcome"] for r in records],deadlock_criterion=records[0]["deadlock_criterion"])
    (folder/f"seed{seed}_video_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(dict(video=str(video),frames=frames,complete=True)))


def main(folder,seed):
    render_pair([folder/f"seed{seed}_{name}.npz" for name in ("goal_cbf","route_witness")],folder,seed)


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--root",type=Path,default=Path("diagnostics/drone_3d_pilot_v3"))
    parser.add_argument("--seed",type=int,default=17)
    parser.add_argument("--baseline",type=Path);parser.add_argument("--intervention",type=Path)
    args=parser.parse_args()
    if args.baseline or args.intervention:
        if not (args.baseline and args.intervention):parser.error("both trace paths required")
        render_pair([args.baseline,args.intervention],args.root,args.seed,pipeline_mode=True)
    else:main(args.root,args.seed)
