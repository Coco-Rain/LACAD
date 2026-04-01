import cadquery as cq

result = (
cq.Workplane("XY")
.box(30, 25, 8)
.union(
cq.Workplane("YZ")
.transformed(offset=(12, 0, 4))
.cylinder(12, 6)
)
)
result
cq.exporters.export(result, 'GT.stl')