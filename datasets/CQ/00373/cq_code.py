import cadquery as cq

result = (
cq.Sketch()
.arc((0,0),1.,0.,360.)
.arc((1,1.5),0.5,0.,360.)
.segment((0.,2),(-1,3.))
.hull()
)
cq.exporters.export(result, 'GT.stl')