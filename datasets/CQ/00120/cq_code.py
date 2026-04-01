import cadquery as cq

base = cq.Workplane("XY").circle(10).extrude(5)
result = (
base.faces(">Z")
.workplane()
.center(0, 0)
.circle(5)
.cutBlind(-2)
)
cq.exporters.export(result, 'GT.stl')