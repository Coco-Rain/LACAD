import cadquery as cq

result = (
cq.Workplane("XY")
.moveTo(2, 2)
.rect(6, 4)
.extrude(2)
.faces(">Z")
.workplane()
.rect(4, 2)
.cutBlind(-1)
)
cq.exporters.export(result, 'GT.stl')