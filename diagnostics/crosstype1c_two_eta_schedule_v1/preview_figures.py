"""Render representative standalone plots for the primary agent's review."""
from core import *
import subprocess,xml.etree.ElementTree as ET
from PIL import Image
def main():
    folder=HERE/'figure_previews';folder.mkdir(exist_ok=True);profile=HERE/'firefox_preview_profile';profile.mkdir(exist_ok=True);config=HERE/'browser_config';config.mkdir(exist_ok=True)
    records=[]
    for name in ['eta_pair_scatter_component0.svg','eta_pair_hist_component0.svg','direction_cosine.svg','switch_times_self_schedule.svg','switch_times_fixed500.svg','critic_curves_Flow.svg']:
        source=HERE/'figures'/name;root=ET.parse(source).getroot();width=int(root.attrib['width']);height=int(root.attrib['height']);target=folder/Path(name).with_suffix('.png');env={**os.environ,'LIBGL_ALWAYS_SOFTWARE':'1','MOZ_WEBRENDER':'0','MOZ_CRASHREPORTER_DISABLE':'1','XDG_CONFIG_HOME':str(config)}
        r=subprocess.run(['/usr/bin/firefox','--headless','--no-remote','--profile',str(profile),'--screenshot',str(target),'--window-size',f'{width},{height}',source.as_uri()],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60);(folder/(name+'.log')).write_text(r.stdout);assert r.returncode==0 and target.exists()
        with Image.open(target) as im:assert im.size==(width,height) and len(im.getcolors(width*height) or [])>5
        records.append(dict(source=str(source),preview=str(target),size=[width,height],preview_sha256=sha(target)))
    write(HERE/'FIGURE_PREVIEWS.json',dict(utc=utc(),previews=records,rendering='Headless Firefox, software graphics; own profile/cache/config'))
    seal('FIGURE_PREVIEWS.receipt.json',[HERE/'FIGURE_PREVIEWS.json',HERE/'preview_figures.py',*[Path(r['preview']) for r in records]])
    print('Rendered six representative previews; manual visual review still required',flush=True)
if __name__=='__main__':main()
