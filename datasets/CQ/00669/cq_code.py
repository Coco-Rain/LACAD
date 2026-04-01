import cadquery as cq

result = (
cq.Workplane("XY")
.box(30, 25, 8)
.faces(">Z")
.workplane()
.ellipse(10, 6)
.cutBlind(-6)
)
cq.exporters.export(result, 'GT.stl')