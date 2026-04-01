import cadquery as cq

result = (
cq.Workplane("XY")
.wedge(10, 5, 5, 0, 0, 0, 2.5)
)
cq.exporters.export(result, 'GT.stl')