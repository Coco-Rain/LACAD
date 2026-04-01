import cadquery as cq

result = (
cq.Workplane("XY")
.sketch()
.rect(4, 2)
)
cq.exporters.export(result, 'GT.stl')