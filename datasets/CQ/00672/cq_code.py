import cadquery as cq

result = (
cq.Sketch()
.ellipse(8, 3)
)
cq.exporters.export(result, 'GT.stl')