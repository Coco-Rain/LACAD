import cadquery as cq

result = (
cq.Workplane("XY")
.box(30, 20, 12)
.cut(
cq.Workplane("XZ")
.transformed(offset=(0, 6, 6))
.cylinder(25, 5)
)
.cut(
cq.Workplane("YZ")
.transformed(offset=(15, 0, 6))
.cylinder(25, 5)
)
)
cq.exporters.export(result, 'GT.stl')