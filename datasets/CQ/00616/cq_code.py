import cadquery as cq

result = (
cq.Workplane("XY")
.moveTo(2, 2)
.rect(6, 4)
.extrude(3)
.faces(">Z")
.circle(1)
.cutBlind(-2)
)
cq.exporters.export(result, 'GT.stl')