import cadquery as cq

result = (
cq.Workplane("XY")
.box(30, 20, 15)
.union(
cq.Workplane("XY")
.transformed(offset=(0, 0, 7.5))
.ellipse(10, 6)
.extrude(12)
)
)
result
cq.exporters.export(result, 'GT.stl')