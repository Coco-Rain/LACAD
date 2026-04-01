import cadquery as cq

result = (
cq.Workplane("XY")
.center(2, 2)
.rect(4, 6)
.extrude(1.5)
.faces(">Z")
.workplane()
.center(-1, -1)
.circle(1)
.cutBlind(-0.5)
)
cq.exporters.export(result, 'GT.stl')