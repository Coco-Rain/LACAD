import cadquery as cq

result = (
cq.Workplane("XY")
.box(30, 25, 12)
.union(
cq.Workplane("XY")
.transformed(offset=(0, 0, 6))
.ellipse(10, 7)
.extrude(8)
)
.edges("|Z").fillet(4)
)
cq.exporters.export(result, 'GT.stl')