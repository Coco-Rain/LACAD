import cadquery as cq

result = (
cq.Workplane()
.box(1,1,1)
.wires('>Z').toPending()
.translate((1,.1,1))
.rotate((0,0,0),(0,0,1),45)
.toPending()
.loft()
)
cq.exporters.export(result, 'GT.stl')