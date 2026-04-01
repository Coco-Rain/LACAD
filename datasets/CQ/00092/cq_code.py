import cadquery as cq

result = (
cq.Workplane("XY")
.box(10, 10, 10)
.intersect(
cq.Workplane("XY")
.transformed(offset=(5, 0, 0))
.circle(3)
.extrude(10)
.val()
)
)
cq.exporters.export(result, 'GT.stl')