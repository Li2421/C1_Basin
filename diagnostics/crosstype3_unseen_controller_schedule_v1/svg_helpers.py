from common import *
import html

def text(x,y,s,size=12,fill='#111',anchor='start'):return f'<text x="{x:.2f}" y="{y:.2f}" font-family="sans-serif" font-size="{size}" fill="{fill}" text-anchor="{anchor}">{html.escape(str(s))}</text>'


def start(title,w=1280,h=650):return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">','<rect width="100%" height="100%" fill="white"/>',text(25,30,title,19)]


def save(path,body):path.write_text('\n'.join([*body,'</svg>'])+'\n')


def line(x1,y1,x2,y2,color='#777',dash=''):return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-dasharray="{dash}"/>'


def axes(body,x,y,w,h,xmin,xmax,ymin,ymax,xlabel,ylabel):
    body.extend([line(x,y,x,y+h),line(x,y+h,x+w,y+h),text(x+w/2,y+h+39,xlabel,12,anchor='middle'),text(x,y-12,ylabel,12)])
    for value in np.linspace(xmin,xmax,5):
        px=x+(value-xmin)/(xmax-xmin)*w;body.extend([line(px,y+h,px,y+h+5),text(px,y+h+20,f'{value:.2f}',10,anchor='middle')])
    for value in np.linspace(ymin,ymax,5):
        py=y+h-(value-ymin)/(ymax-ymin)*h;body.extend([line(x-5,py,x,py),text(x-9,py+4,f'{value:.2f}',10,anchor='end')])
    return lambda a,b:(x+(a-xmin)/(xmax-xmin)*w,y+h-(b-ymin)/(ymax-ymin)*h)

