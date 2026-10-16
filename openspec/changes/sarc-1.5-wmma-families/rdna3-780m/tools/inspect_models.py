import json
import mmap
import pathlib
import struct
import sys
from flatbuffers.table import Table
from flatbuffers import number_types as nt

class FB:
    def __init__(self, data, pos):
        self.t = Table(data, pos)
    def off(self, field):
        return self.t.Offset(4 + 2 * field)
    def scalar(self, field, flags=nt.Uint32Flags, default=0):
        o = self.off(field)
        return self.t.Get(flags, self.t.Pos + o) if o else default
    def table(self, field):
        o = self.off(field)
        return FB(self.t.Bytes, self.t.Indirect(self.t.Pos + o)) if o else None
    def string(self, field):
        o = self.off(field)
        return self.t.String(self.t.Pos + o).decode() if o else ''
    def tables(self, field):
        o = self.off(field)
        return [FB(self.t.Bytes, self.t.Indirect(self.t.Vector(o) + 4*i))
                for i in range(self.t.VectorLen(o))] if o else []
    def vector(self, field, fmt='i'):
        o = self.off(field)
        if not o: return []
        return list(struct.unpack_from('<'+fmt*self.t.VectorLen(o), self.t.Bytes, self.t.Vector(o)))

def value(v, vulkan=False):
    kind=v.scalar(0,nt.Uint8Flags)
    t=v.table(1)
    if kind==1: return None
    if kind==2: return t.scalar(0,nt.Int64Flags)
    if kind==5:
        return {'dtype':t.scalar(0,nt.Int8Flags),'dims':t.vector(1 if vulkan else 2,'I' if vulkan else 'i'),
                'storage':t.scalar(4,nt.Uint8Flags,255) if vulkan else None,
                'layout':t.scalar(5,nt.Uint8Flags,255) if vulkan else None}
    if (kind==3 and not vulkan) or (kind==4 and vulkan): return bool(t.scalar(0,nt.Uint8Flags))
    return {'kind':kind}

root=pathlib.Path(sys.argv[1])
out=[]
for model,stem in [('llama-3.2-1b','llama3_2-1b'),('llama-3.2-3b','llama3_2-3b'),('llama-3.1-8b','llama3_1-8b')]:
    for scheme in ('4w','8da4w'):
        path=root/model/'exported'/f'{stem}_vulkan_{scheme}.pte'
        with path.open('rb') as f, mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as b:
            assert b[4:8]==b'ET12' and b[8:12]==b'eh00'
            segment_base=struct.unpack_from('<Q',b,24)[0]
            program=FB(b,struct.unpack_from('<I',b)[0])
            segments=program.tables(4)
            row={'path':str(path),'bytes':len(b),'constant_methods':{},'operators':[],
                 'linear_configurations':[],'embeddings':[],'graph_overrides':[]}
            for plan in program.tables(1):
                name=plan.string(0)
                vals=[value(v) for v in plan.tables(2)]
                if name!='forward':
                    row['constant_methods'][name]=[vals[i] for i in plan.vector(4)]
                    continue
                row['operators']=[o.string(0)+'.'+o.string(1) for o in plan.tables(6)]
                row['possible_embedding_tensors']=[v for v in vals if isinstance(v,dict) and v.get('dims') and 128256 in v['dims']]
                for delegate in plan.tables(7):
                    assert delegate.string(0)=='VulkanBackend'
                    ref=delegate.table(1)
                    loc=ref.scalar(0,nt.Uint8Flags); ix=ref.scalar(1)
                    if loc==1: start=segment_base+segments[ix].scalar(0,nt.Uint64Flags)
                    else:
                        inline=program.tables(3)[ix]
                        start=inline.t.Vector(inline.off(0))
                    assert b[start+4:start+8]==b'VH00'
                    graph_start=start+struct.unpack_from('<I',b,start+10)[0]
                    assert b[graph_start+4:graph_start+8]==b'VK00'
                    graph=FB(b,graph_start+struct.unpack_from('<I',b,graph_start)[0])
                    vv=[value(v,True) for v in graph.tables(2)]
                    row['graph_overrides'].append({'storage':graph.scalar(7,nt.Uint8Flags,255),'layout':graph.scalar(8,nt.Uint8Flags,255)})
                    for op in graph.tables(1):
                        op_name=op.string(1)
                        if 'linear' not in op_name and 'embedding' not in op_name: continue
                        args=[vv[i] for i in op.vector(2)]
                        entry={'name':op_name,'args':args}
                        dest='embeddings' if 'embedding' in op_name else 'linear_configurations'
                        if entry not in row[dest]:row[dest].append(entry)
            out.append(row)
print(json.dumps(out,indent=2))
