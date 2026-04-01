import cadquery as cq

result = (
cq.Sketch()
.slot(12, 4, 45)
)
cq.exporters.export(result, 'GT.stl')