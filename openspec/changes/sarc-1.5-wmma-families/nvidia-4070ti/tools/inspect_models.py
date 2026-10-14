import json,mmap,struct,sys
from pathlib import Path

class Flatbuffer:
    def __init__(self,data,base=0):
        self.data=data
        self.base=base
        self.root=self.pointer(base)
    def number(self,pos,fmt='I'):
        return struct.unpack_from('<'+fmt,self.data,pos)[0]
    def field(self,table,index):
        vtable=table-self.number(table,'i')
        off=4+2*index
        return table+self.number(vtable+off,'H') if off < self.number(vtable,'H') and self.number(vtable+off,'H') else None
    def scalar(self,table,index,fmt='I',default=0):
        pos=self.field(table,index)
        return self.number(pos,fmt) if pos is not None else default
    def pointer(self,pos):
        return pos+self.number(pos)
    def target(self,table,index):
        pos=self.field(table,index)
        return self.pointer(pos) if pos is not None else None
    def string(self,table,index):
        pos=self.target(table,index)
        return self.data[pos+4:pos+4+self.number(pos)].decode() if pos is not None else ''
    def vector(self,table,index,fmt=None):
        pos=self.target(table,index)
        if pos is None:return []
        return [self.number(pos+4+i*struct.calcsize('<'+fmt),fmt) if fmt else self.pointer(pos+4+4*i) for i in range(self.number(pos))]
    def value(self,table):
        kind=self.scalar(table,0,'B')
        value=self.target(table,1)
        if kind==2:return self.scalar(value,0,'q')
        if kind==4:return bool(self.scalar(value,0,'B'))
        if kind==5:return {'datatype':self.scalar(value,0,'b'),'dims':self.vector(value,1,'I'),'storage':self.scalar(value,4,'B',255),'layout':self.scalar(value,5,'B',255),'constant':self.scalar(value,2,'i')}
        return {'kind':kind}

results=[]
for path in sys.argv[1:]:
    with open(path,'rb') as file,mmap.mmap(file.fileno(),0,access=mmap.ACCESS_READ) as data:
        assert data[4:8]==b'ET12' and data[8:12]==b'eh00'
        f=Flatbuffer(data)
        methods={}
        graphs=[]
        segments=f.vector(f.root,4)
        for method in f.vector(f.root,1):
            name=f.string(method,0)
            vals=f.vector(method,2)
            if name!='forward':
                assert not f.vector(method,3)
                assert all(not f.vector(chain,2) for chain in f.vector(method,5))
                output=f.vector(method,4,'i')
                result=[]
                for index in output:
                    v=vals[index]
                    kind=f.scalar(v,0,'B');p=f.target(v,1)
                    assert kind in (2,3)
                    result.append(f.scalar(p,0,'q' if kind==2 else 'B'))
                methods[name]=result
            for delegate in f.vector(method,7):
                name=f.string(delegate,0)
                ref=f.target(delegate,1)
                assert f.scalar(ref,0,'B')==1
                segment=segments[f.scalar(ref,1)]
                start=f.number(24,'Q')+f.scalar(segment,0,'Q')
                assert data[start+4:start+8]==b'VH00'
                graphbase=start+f.number(start+10,'I')
                assert data[graphbase+4:graphbase+8]==b'VK00'
                g=Flatbuffer(data,graphbase)
                values=[g.value(v) for v in g.vector(g.root,2)]
                calls=[]
                for call in g.vector(g.root,1):
                    opname=g.string(call,1)
                    if any(key in opname for key in ['q4gsw','embedding']):
                        args=g.vector(call,2,'i')
                        calls.append({'name':opname,'args':[values[a] for a in args]})
                graphs.append({'backend':name,'storage_override':g.scalar(g.root,7,'B',255),'layout_override':g.scalar(g.root,8,'B',255),'calls':calls})
        results.append({'path':path,'bytes':len(data),'constant_methods_static':methods,'graphs':graphs})
print(json.dumps(results,indent=2))
