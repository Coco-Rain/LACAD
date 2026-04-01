import cadquery as cq

result = (
cq.Workplane("XY")
.rect(8, 8)
.extrude(2)
.faces(">Z")
.circle(3)
.cutBlind(-1)
)
cq.exporters.export(result, 'GT.stl')