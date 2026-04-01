import cadquery as cq

result = (
cq.Sketch()
.trapezoid(4,3,90)
.rarray(.6,1,5,1)
.slot(1.5,0.4, mode='s', angle=90)
)
cq.exporters.export(result, 'GT.stl')