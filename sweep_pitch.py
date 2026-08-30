#!/usr/bin/env python3
"""Map a fine probe sweep to relative (fan-subtracted) tonal peaks."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks, stft

def load(path):
    rate, data = wavfile.read(path)
    scale = np.iinfo(data.dtype).max if np.issubdtype(data.dtype, np.integer) else 1.0
    x = data.astype(float) / scale
    return rate, x.mean(axis=1) if x.ndim == 2 else x

def main():
    p=argparse.ArgumentParser(description='按 sweep.json 将录音切段并扣除 fan.wav 固定频谱')
    p.add_argument('recording', type=Path); p.add_argument('fan_reference', type=Path); p.add_argument('manifest', type=Path)
    p.add_argument('-o','--output',type=Path,default=Path('sweep_relative_pitch.csv'))
    p.add_argument('--recording-offset',type=float,default=None,help='第一个定位音在录音中的秒数；默认自动搜索')
    p.add_argument('--top',type=int,default=5); p.add_argument('--fft-size',type=int,default=8192)
    a=p.parse_args(); rate,x=load(a.recording); rr,fan=load(a.fan_reference)
    if rate!=rr: p.error('录音与 fan.wav 采样率不同')
    j=json.loads(a.manifest.read_text()); seg=j['segments']; interval=float(j.get('duration_s',1)+j.get('rest_s',.5))
    f,_,zf=stft(fan,fs=rate,nperseg=a.fft_size,noverlap=a.fft_size*3//4,boundary='zeros',padded=True)
    fan_db=np.median(20*np.log10(abs(zf)+1e-10),axis=1)
    if a.recording_offset is None:
        # Locate the regular 1.5 s marker comb near 1760 Hz.
        _,tm,zz=stft(x,fs=rate,nperseg=4096,noverlap=3072,boundary='zeros',padded=True)
        marker=np.abs(zz[np.argmin(abs(_-1760))]); step=tm[1]-tm[0]; best=(-1,None)
        for t0 in np.arange(0,min(interval,3),step):
            idx=np.clip(np.round((t0+np.arange(len(seg))*interval)/step).astype(int),0,len(marker)-1)
            score=float(np.sum(marker[idx]))
            if score>best[0]: best=(score,t0)
        offset=float(best[1]); print(f'自动定位：第一个标记约在录音 {offset:.3f}s')
    else: offset=a.recording_offset
    rows=[]
    for n,item in enumerate(seg):
        start=offset+float(j.get('rest_s',.5))+n*interval; chunk=x[int(start*rate):int((start+float(j.get('duration_s',1)))*rate)]
        if len(chunk)<a.fft_size//2: continue
        _,_,z=stft(chunk,fs=rate,nperseg=a.fft_size,noverlap=a.fft_size*3//4,boundary='zeros',padded=True)
        target=np.median(20*np.log10(abs(z)+1e-10),axis=1); relative=target-fan_db
        band=(f>=100)&(f<=12000); vals=relative[band]; freqs=f[band]; local=np.convolve(vals,np.ones(9)/9,'same'); prom=vals-local
        peaks,_=find_peaks(prom,height=1.0,distance=max(2,int(30/(rate/a.fft_size))))
        chosen=peaks[np.argsort(prom[peaks])[-a.top:]][::-1] if len(peaks) else []
        for rank,k in enumerate(chosen,1): rows.append({'index':n,'size':item['size'],'rank':rank,'frequency_hz':round(float(freqs[k]),2),'relative_prominence_db':round(float(prom[k]),2),'segment_start_s':round(start,3)})
    with a.output.open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=['index','size','rank','frequency_hz','relative_prominence_db','segment_start_s']); w.writeheader();w.writerows(rows)
    print(f'写入 {len(rows)} 个候选：{a.output}')
if __name__=='__main__': main()
