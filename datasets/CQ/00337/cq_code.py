import cadquery as cq

result = (
cq.Workplane("XY")
.box(12, 12, 8)
.faces(">Z")
.workplane()
.rect(8, 8)
.cutBlind(-6)
.shells()
.faces(">Z")
.workplane()
.circle(1.5)
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')