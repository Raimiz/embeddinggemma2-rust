import math, wave
import numpy as np
from PIL import Image

def aspect_size(h,w,patch,max_soft,pool):
    target_px=max_soft*pool*pool*patch*patch
    side=pool*patch
    factor=math.sqrt(target_px/(h*w))
    th=math.floor(factor*h/side)*side; tw=math.floor(factor*w/side)*side
    max_side=(max_soft*pool*pool//(pool*pool))*side
    if th==0 and tw==0: raise ValueError("zero resize")
    if th==0: th=side; tw=min(math.floor(w/h)*side,max_side)
    elif tw==0: tw=side; th=min(math.floor(h/w)*side,max_side)
    return th,tw

def _coeff(in_size,out_size):
    scale=in_size/out_size; fs=max(scale,1.0); support=2*fs; ksize=math.ceil(support)*2+1
    bounds=[]; weights=[]; wt_max=0.
    for xx in range(out_size):
        center=(xx+0.5)*scale; ss=1/fs
        xmin=max(0,int(center-support+0.5)); xmax=max(0,min(in_size,int(center+support+0.5)))
        ks=[]; ww=0.
        for i in range(xmax-xmin):
            x=abs((i+xmin-center+0.5)*ss)
            if x<1: v=(((-0.5+2)*x-(-0.5+3))*x*x+1)
            elif x<2: v=(((x-5)*x+8)*x-4)*(-0.5)
            else: v=0.
            ks.append(v); ww+=v
        if ww: ks=[v/ww for v in ks]
        wt_max=max(wt_max,max(ks,default=0.))
        bounds.append((xmin,xmax-xmin)); weights.append(ks)
    p=0
    while p<22 and int(.5+wt_max*(1<<(p+1))) < (1<<15): p+=1
    fixed=[]
    for ks in weights:
        fixed.append([int(v*(1<<p)+(0.5 if v>=0 else -0.5)) for v in ks])
    return bounds,fixed,p

def _resize_axis(src,axis,out_size):
    in_size=src.shape[axis]; bounds,weights,p=_coeff(in_size,out_size)
    ksize=max(map(len,weights)); idx=np.empty((out_size,ksize),dtype=np.int64); coeff=np.zeros((out_size,ksize),dtype=np.int64)
    for j,((lo,n),ww) in enumerate(zip(bounds,weights)):
        idx[j]=np.clip(lo+np.arange(ksize),0,in_size-1); coeff[j,:n]=ww
    vals=np.take(src,idx,axis=axis).astype(np.int64)
    if axis==1:
        # [H, output, kernel, C]
        acc=(vals*coeff[None,:,:,None]).sum(axis=2,dtype=np.int64)+(1<<(p-1))
    else:
        # [output, kernel, W, C]
        acc=(vals*coeff[:,:,None,None]).sum(axis=1,dtype=np.int64)+(1<<(p-1))
    return np.clip(acc>>p,0,255).astype(np.uint8)

def image_inputs(path, max_soft_tokens=280, patch_size=16, pool=3):
    img=np.asarray(Image.open(path).convert("RGB"),dtype=np.uint8)
    h,w=img.shape[:2]; th,tw=aspect_size(h,w,patch_size,max_soft_tokens,pool)
    if (h,w)!=(th,tw):
        img=_resize_axis(img,1,tw); img=_resize_axis(img,0,th)
    gh,gw=th//patch_size,tw//patch_size
    # row-major [patch_y,patch_x,patch_y_pixel,patch_x_pixel,channel]
    patches=img.reshape(gh,patch_size,gw,patch_size,3).transpose(0,2,1,3,4).reshape(1,gh*gw,-1).astype(np.float32)*(1/255)
    pos=np.stack(np.meshgrid(np.arange(gw,dtype=np.int64),np.arange(gh,dtype=np.int64)),axis=-1).reshape(1,-1,2)
    return patches,pos

def audio_inputs(path, cfg):
    with wave.open(str(path),"rb") as wf:
        if wf.getnchannels()!=1 or wf.getframerate()!=int(cfg["sampling_rate"]) or wf.getsampwidth()!=2: raise ValueError("expected mono PCM16 at configured sample rate")
        wav=np.frombuffer(wf.readframes(wf.getnframes()),dtype="<i2").astype(np.float32)/32768.0
    fl=int(cfg["frame_length"]); hop=int(cfg["hop_length"]); nfft=int(cfg["fft_length"]); pad=fl//2
    frames=(len(wav)+pad-(fl+1))//hop+1
    x=np.pad(wav,(pad,pad+fl+hop),mode="constant")
    window=(.5-.5*np.cos(2*np.pi*np.arange(fl)/fl)).astype(np.float32)
    # HTK mel filters, matching the upstream front end.
    nmel=int(cfg["feature_size"]); sr=int(cfg["sampling_rate"])
    hz2mel=lambda hz:2595*np.log10(1+hz/700)
    mel2hz=lambda m:700*(10**(m/2595)-1)
    freqs=mel2hz(np.linspace(hz2mel(float(cfg["min_frequency"])),hz2mel(float(cfg["max_frequency"])),nmel+2))
    fftfreq=np.linspace(0,sr/2,nfft//2+1)
    delta=np.diff(freqs); filters=np.maximum(0,np.minimum(-(freqs[:-2,None]-fftfreq[None,:]).T/delta[:-1],(freqs[2:,None]-fftfreq[None,:]).T/delta[1:]))
    starts=np.arange(frames)*hop
    windows=np.stack([x[s:s+fl] for s in starts])
    mag=np.abs(np.fft.rfft(windows*window[None,:],n=nfft,axis=-1))
    mel=mag@filters
    feats=np.log(mel+float(cfg["mel_floor"])).astype(np.float32)[None,:,:]
    mask=np.ones(feats.shape[:2],dtype=np.bool_)
    return feats,mask
