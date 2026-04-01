import cadquery as cq

result = (
cq.Workplane("XY")
.wedge(20, 10, 10, 0, 0, 0, 5)
)
cq.exporters.export(result, 'GT.stl')