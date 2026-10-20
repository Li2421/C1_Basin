from core import *
import subprocess
from PIL import Image
import xml.etree.ElementTree as ET


def main():
    folder=HERE/'figure_previews';folder.mkdir(exist_ok=True);profile=HERE/'firefox_preview_profile';profile.mkdir(exist_ok=True)
    names=['action_distribution_distributed_cross_entropy.svg','locked_trajectory_centralized_mppi.svg','scaling_m.svg','scaling_repair_class_count.svg','critic_curves_Flow.svg']
    records=[]
    for name in names:
        source=HERE/'figures'/name;root=ET.parse(source).getroot();width=int(root.attrib['width']);height=int(root.attrib['height']);target=folder/Path(name).with_suffix('.png')
        render_env={**os.environ,'LIBGL_ALWAYS_SOFTWARE':'1','MOZ_WEBRENDER':'0','MOZ_CRASHREPORTER_DISABLE':'1'}
        result=subprocess.run(['/usr/bin/firefox','--headless','--no-remote','--profile',str(profile),'--screenshot',str(target),'--window-size',f'{width},{height}',source.as_uri()],env=render_env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
        (folder/(name+'.log')).write_text(result.stdout);assert result.returncode==0 and target.exists()
        with Image.open(target) as im:assert im.size==(width,height);assert len(im.getcolors(width*height) or [])>5
        records.append(dict(source=str(source),preview=str(target),preview_sha256=sha(target),size=[width,height]))
    write(HERE/'FIGURE_PREVIEWS.json',dict(utc=utc(),previews=records,rendering='Headless Firefox; profile/cache/output contained in experiment directory'))
    seal('FIGURE_PREVIEWS.receipt.json',[HERE/'FIGURE_PREVIEWS.json',HERE/'preview_figures.py',*[Path(r['preview']) for r in records]])
    print('Rendered five representative previews for visual inspection')


if __name__=='__main__':main()
