import cadquery as cq

result = (
cq.Workplane("XY")
.moveTo(2, 2)
.rect(4, 6)
.extrude(3)
.faces(">Z")
.workplane()
.circle(1)
.cutBlind(-1)
)
cq.exporters.export(result, 'GT.stl')