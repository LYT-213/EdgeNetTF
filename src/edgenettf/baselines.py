import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CHANNELS = 12
NUM_CLASSES = 49
DROPOUT = 0.3

class CNN1D(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.time_stem = nn.Sequential(
            nn.Conv1d(12,64,7,padding=3), nn.BatchNorm1d(64), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(64,128,7,padding=3), nn.BatchNorm1d(128), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(128,128,5,padding=2), nn.BatchNorm1d(128), nn.GELU(), nn.AdaptiveAvgPool1d(1))
        self.classifier = nn.Sequential(nn.Linear(128,256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(256,num_classes))
    def forward(self,x): return self.classifier(self.time_stem(x).squeeze(-1))

class BasicBlock1D(nn.Module):
    def __init__(self, inc, outc, stride=1):
        super().__init__()
        self.c1=nn.Conv1d(inc,outc,7,stride=stride,padding=3,bias=False); self.b1=nn.BatchNorm1d(outc)
        self.c2=nn.Conv1d(outc,outc,5,padding=2,bias=False); self.b2=nn.BatchNorm1d(outc); self.a=nn.GELU()
        self.ds=nn.Identity() if stride==1 and inc==outc else nn.Sequential(nn.Conv1d(inc,outc,1,stride=stride,bias=False),nn.BatchNorm1d(outc))
    def forward(self,x):
        y=self.a(self.b1(self.c1(x))); y=self.b2(self.c2(y)); return self.a(y+self.ds(x))
class ResNet1D(nn.Module):
    def __init__(self,num_classes=NUM_CLASSES):
        super().__init__(); self.stem=nn.Sequential(nn.Conv1d(12,64,7,padding=3,bias=False),nn.BatchNorm1d(64),nn.GELU(),nn.MaxPool1d(2))
        self.l1=nn.Sequential(BasicBlock1D(64,64),BasicBlock1D(64,64)); self.l2=nn.Sequential(BasicBlock1D(64,128,2),BasicBlock1D(128,128)); self.l3=nn.Sequential(BasicBlock1D(128,256,2),BasicBlock1D(256,256))
        self.pool=nn.AdaptiveAvgPool1d(1); self.classifier=nn.Sequential(nn.Linear(256,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(DROPOUT),nn.Linear(256,num_classes))
    def forward(self,x): x=self.stem(x); x=self.l1(x); x=self.l2(x); x=self.l3(x); return self.classifier(self.pool(x).squeeze(-1))

class InvertedResidual1D(nn.Module):
    def __init__(self,inc,outc,stride,expand):
        super().__init__(); hid=int(round(inc*expand)); self.res=stride==1 and inc==outc; L=[]
        if expand!=1: L += [nn.Conv1d(inc,hid,1,bias=False),nn.BatchNorm1d(hid),nn.ReLU6(inplace=True)]
        L += [nn.Conv1d(hid,hid,3,stride=stride,padding=1,groups=hid,bias=False),nn.BatchNorm1d(hid),nn.ReLU6(inplace=True),nn.Conv1d(hid,outc,1,bias=False),nn.BatchNorm1d(outc)]
        self.block=nn.Sequential(*L)
    def forward(self,x):
        y=self.block(x); return x+y if self.res else y
class MobileNetV2_1D(nn.Module):
    def __init__(self,num_classes=NUM_CLASSES):
        super().__init__(); settings=[(1,16,1,1),(6,24,2,2),(6,32,3,2),(6,64,3,2),(6,96,2,1),(6,128,2,1)]
        L=[nn.Conv1d(12,32,3,stride=2,padding=1,bias=False),nn.BatchNorm1d(32),nn.ReLU6(inplace=True)]; inc=32
        for e,c,n,s in settings:
            for i in range(n): L.append(InvertedResidual1D(inc,c,s if i==0 else 1,e)); inc=c
        L += [nn.Conv1d(inc,256,1,bias=False),nn.BatchNorm1d(256),nn.ReLU6(inplace=True),nn.AdaptiveAvgPool1d(1)]
        self.features=nn.Sequential(*L); self.classifier=nn.Sequential(nn.Linear(256,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(DROPOUT),nn.Linear(256,num_classes))
    def forward(self,x): return self.classifier(self.features(x).squeeze(-1))

class FireModule1D(nn.Module):
    def __init__(self,inc,sq,ex):
        super().__init__(); self.s=nn.Sequential(nn.Conv1d(inc,sq,1,bias=False),nn.BatchNorm1d(sq),nn.GELU()); self.e1=nn.Sequential(nn.Conv1d(sq,ex,1,bias=False),nn.BatchNorm1d(ex),nn.GELU()); self.e3=nn.Sequential(nn.Conv1d(sq,ex,3,padding=1,bias=False),nn.BatchNorm1d(ex),nn.GELU())
    def forward(self,x): x=self.s(x); return torch.cat([self.e1(x),self.e3(x)],1)
class SqueezeNet1D(nn.Module):
    def __init__(self,num_classes=NUM_CLASSES):
        super().__init__(); self.features=nn.Sequential(nn.Conv1d(12,64,7,stride=2,padding=3,bias=False),nn.BatchNorm1d(64),nn.GELU(),nn.MaxPool1d(3,2,1),FireModule1D(64,16,64),FireModule1D(128,16,64),nn.MaxPool1d(3,2,1),FireModule1D(128,32,128),FireModule1D(256,32,128),nn.MaxPool1d(3,2,1),FireModule1D(256,48,192),FireModule1D(384,48,192),nn.AdaptiveAvgPool1d(1)); self.classifier=nn.Sequential(nn.Linear(384,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(DROPOUT),nn.Linear(256,num_classes))
    def forward(self,x): return self.classifier(self.features(x).squeeze(-1))

def channel_shuffle_1d(x,groups=2):
    b,c,l=x.shape; x=x.view(b,groups,c//groups,l).transpose(1,2).contiguous(); return x.view(b,c,l)
class ShuffleUnit1D(nn.Module):
    def __init__(self,inc,outc,stride):
        super().__init__(); self.stride=stride; b=outc//2
        if stride==1:
            self.b2=nn.Sequential(nn.Conv1d(b,b,1,bias=False),nn.BatchNorm1d(b),nn.GELU(),nn.Conv1d(b,b,3,padding=1,groups=b,bias=False),nn.BatchNorm1d(b),nn.Conv1d(b,b,1,bias=False),nn.BatchNorm1d(b),nn.GELU())
        else:
            self.b1=nn.Sequential(nn.Conv1d(inc,inc,3,stride=2,padding=1,groups=inc,bias=False),nn.BatchNorm1d(inc),nn.Conv1d(inc,b,1,bias=False),nn.BatchNorm1d(b),nn.GELU())
            self.b2=nn.Sequential(nn.Conv1d(inc,b,1,bias=False),nn.BatchNorm1d(b),nn.GELU(),nn.Conv1d(b,b,3,stride=2,padding=1,groups=b,bias=False),nn.BatchNorm1d(b),nn.Conv1d(b,b,1,bias=False),nn.BatchNorm1d(b),nn.GELU())
    def forward(self,x):
        if self.stride==1:
            x1,x2=x.chunk(2,1); y=torch.cat([x1,self.b2(x2)],1)
        else: y=torch.cat([self.b1(x),self.b2(x)],1)
        return channel_shuffle_1d(y)
class ShuffleNetV2_1D(nn.Module):
    def __init__(self,num_classes=NUM_CLASSES):
        super().__init__(); self.stem=nn.Sequential(nn.Conv1d(12,48,3,stride=2,padding=1,bias=False),nn.BatchNorm1d(48),nn.GELU(),nn.MaxPool1d(3,2,1)); self.s2=nn.Sequential(ShuffleUnit1D(48,96,2),ShuffleUnit1D(96,96,1),ShuffleUnit1D(96,96,1)); self.s3=nn.Sequential(ShuffleUnit1D(96,192,2),ShuffleUnit1D(192,192,1),ShuffleUnit1D(192,192,1),ShuffleUnit1D(192,192,1)); self.s4=nn.Sequential(ShuffleUnit1D(192,384,2),ShuffleUnit1D(384,384,1),ShuffleUnit1D(384,384,1)); self.c5=nn.Sequential(nn.Conv1d(384,512,1,bias=False),nn.BatchNorm1d(512),nn.GELU(),nn.AdaptiveAvgPool1d(1)); self.classifier=nn.Sequential(nn.Linear(512,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(DROPOUT),nn.Linear(256,num_classes))
    def forward(self,x): x=self.stem(x); x=self.s2(x); x=self.s3(x); x=self.s4(x); return self.classifier(self.c5(x).squeeze(-1))

class DenseLayer1D(nn.Module):
    def __init__(self,inc,g,bn_size=4,drop=0.1):
        super().__init__(); mid=bn_size*g; self.net=nn.Sequential(nn.BatchNorm1d(inc),nn.GELU(),nn.Conv1d(inc,mid,1,bias=False),nn.BatchNorm1d(mid),nn.GELU(),nn.Conv1d(mid,g,3,padding=1,bias=False)); self.drop=drop
    def forward(self,x):
        y=self.net(x); y=F.dropout(y,p=self.drop,training=self.training) if self.drop>0 else y; return torch.cat([x,y],1)
class DenseBlock1D(nn.Module):
    def __init__(self,n,inc,g):
        super().__init__(); L=[]; c=inc
        for _ in range(n): L.append(DenseLayer1D(c,g)); c+=g
        self.block=nn.Sequential(*L); self.out_channels=c
    def forward(self,x): return self.block(x)
class Transition1D(nn.Module):
    def __init__(self,inc,outc): super().__init__(); self.net=nn.Sequential(nn.BatchNorm1d(inc),nn.GELU(),nn.Conv1d(inc,outc,1,bias=False),nn.AvgPool1d(2,2))
    def forward(self,x): return self.net(x)
class DenseNet1D(nn.Module):
    def __init__(self,num_classes=NUM_CLASSES,growth_rate=32,block_layers=(4,6,4)):
        super().__init__(); c=64; self.stem=nn.Sequential(nn.Conv1d(12,c,7,stride=2,padding=3,bias=False),nn.BatchNorm1d(c),nn.GELU(),nn.MaxPool1d(3,2,1)); self.b1=DenseBlock1D(block_layers[0],c,growth_rate); c=self.b1.out_channels; self.t1=Transition1D(c,c//2); c//=2; self.b2=DenseBlock1D(block_layers[1],c,growth_rate); c=self.b2.out_channels; self.t2=Transition1D(c,c//2); c//=2; self.b3=DenseBlock1D(block_layers[2],c,growth_rate); c=self.b3.out_channels; self.norm=nn.BatchNorm1d(c); self.act=nn.GELU(); self.pool=nn.AdaptiveAvgPool1d(1); self.classifier=nn.Sequential(nn.Linear(c,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(DROPOUT),nn.Linear(256,num_classes))
    def forward(self,x): x=self.stem(x); x=self.b1(x); x=self.t1(x); x=self.b2(x); x=self.t2(x); x=self.b3(x); return self.classifier(self.pool(self.act(self.norm(x))).squeeze(-1))

class InceptionModule1D(nn.Module):
    def __init__(self,inc,bottleneck=32,outc=32,kernels=(9,19,39)):
        super().__init__(); self.bottleneck=nn.Sequential(nn.Conv1d(inc,bottleneck,1,bias=False),nn.BatchNorm1d(bottleneck),nn.GELU()) if inc>1 else nn.Identity(); cin=bottleneck if inc>1 else inc; self.convs=nn.ModuleList([nn.Conv1d(cin,outc,k,padding=k//2,bias=False) for k in kernels]); self.pool=nn.Sequential(nn.MaxPool1d(3,1,1),nn.Conv1d(inc,outc,1,bias=False)); self.bn=nn.BatchNorm1d(outc*(len(kernels)+1)); self.act=nn.GELU(); self.out_channels=outc*(len(kernels)+1)
    def forward(self,x): b=self.bottleneck(x); y=torch.cat([*(c(b) for c in self.convs),self.pool(x)],1); return self.act(self.bn(y))
class InceptionBlock1D(nn.Module):
    def __init__(self,inc):
        super().__init__(); self.inc=InceptionModule1D(inc); self.out_channels=self.inc.out_channels; self.res=nn.Identity() if inc==self.out_channels else nn.Sequential(nn.Conv1d(inc,self.out_channels,1,bias=False),nn.BatchNorm1d(self.out_channels)); self.act=nn.GELU()
    def forward(self,x): return self.act(self.inc(x)+self.res(x))
class InceptionTime1D(nn.Module):
    def __init__(self,num_classes=NUM_CLASSES,num_blocks=6):
        super().__init__(); L=[]; c=12
        for _ in range(num_blocks): b=InceptionBlock1D(c); L.append(b); c=b.out_channels
        self.blocks=nn.Sequential(*L); self.classifier=nn.Sequential(nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(c,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(DROPOUT),nn.Linear(256,num_classes))
    def forward(self,x): return self.classifier(self.blocks(x))
