import cadquery as cq

result = (
cq.Workplane("XY")
.cylinder(12, 6)
.union(
cq.Workplane("XZ")
.transformed(offset=(0, 8, 0))
.sphere(6)
)
)
result
cq.exporters.export(result, 'GT.stl')