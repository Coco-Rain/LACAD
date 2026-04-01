import cadquery as cq

result = (
cq.Workplane("XY")
.rect(20, 10)
.extrude(5)
.faces(">Z")
.workplane()
.rect(10, 5)
.cutBlind(-2)
)
cq.exporters.export(result, 'GT.stl')