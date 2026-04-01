import cadquery as cq

result = (cq.Sketch()
.circle(2)
.edges()
.distribute(20)
.circle(1)
)
cq.exporters.export(result, 'GT.stl')