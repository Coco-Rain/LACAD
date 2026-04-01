import cadquery as cq

result = (
cq.Workplane("XY")
.box(20, 15, 5)
)
cq.exporters.export(result, 'GT.stl')