import cadquery as cq

result = (
cq.Workplane("XY")
.center(15, 15)
.rect(8, 8)
.extrude(5)
.faces(">Z")
.workplane()
.polygon(6, 4)
.cutBlind(-3)
)
cq.exporters.export(result, 'GT.stl')