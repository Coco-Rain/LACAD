import cadquery as cq

result = (
cq.Workplane("XY")
.box(20, 20, 2)
.faces(">Z")
.workplane()
.center(-5, -5)
.rect(4, 4)
.cutBlind(-2)
)
cq.exporters.export(result, 'GT.stl')