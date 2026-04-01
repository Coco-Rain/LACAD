import cadquery as cq

result = (
cq.Workplane("XY")
.box(30, 25, 12)
.union(
cq.Workplane("XY")
.transformed(offset=(0, 0, 6))
.cylinder(10, 5)
)
.edges("|Z")
.fillet(3)
)
cq.exporters.export(result, 'GT.stl')