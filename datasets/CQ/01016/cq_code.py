import cadquery as cq

result = (
cq.Workplane("XY")
.box(12, 8, 2)
.faces(">Z")
.workplane()
.spline([(-4, -2), (0, 3), (4, -2)])
.close()
.cutBlind(-1.5)
)
cq.exporters.export(result, 'GT.stl')