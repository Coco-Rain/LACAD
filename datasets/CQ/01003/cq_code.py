import cadquery as cq

result = (
cq.Workplane("XY")
.box(8, 8, 0.5)
.faces(">Z")
.workplane()
.rect(6, 6)
.cutBlind(-0.25)
)
cq.exporters.export(result, 'GT.stl')