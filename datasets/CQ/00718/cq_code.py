import cadquery as cq

result = (
cq.Sketch()
.segment((0,0), (0,3.),"s1")
.arc((0.,3.), (1.5,1.5), (0.,0.),"a1")
.assemble()
)
cq.exporters.export(result, 'GT.stl')