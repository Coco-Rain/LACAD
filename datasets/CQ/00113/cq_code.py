import cadquery as cq

result = (cq.Sketch()
.circle(2)
.edges()
.distribute(20)
.rect(1,2)
)
cq.exporters.export(result, 'GT.stl')