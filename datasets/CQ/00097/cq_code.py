import cadquery as cq

result = (
cq.Workplane("XY")
.sphere(5)
)
cq.exporters.export(result, 'GT.stl')