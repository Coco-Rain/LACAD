import cadquery as cq

result = (
cq.Workplane("XY")
.rect(30, 20)
.extrude(8)
.faces(">Z")
.workplane()
.ellipse(10, 6)
.cutBlind(-5)
)
cq.exporters.export(result, 'GT.stl')