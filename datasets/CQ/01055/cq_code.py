import cadquery as cq

result = (
cq.Workplane("XY")
.box(10, 20, 5)
)
cq.exporters.export(result, 'GT.stl')