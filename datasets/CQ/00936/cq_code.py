import cadquery as cq

result = (
cq.Workplane("XY")
.rect(10, 5)
.extrude(3)
.faces(">Z")
.workplane()
.hole(2)
.faces("<Z")
.workplane()
.rect(4, 2)
.cutBlind(-1)
)
cq.exporters.export(result, 'GT.stl')