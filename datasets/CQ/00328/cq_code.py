import cadquery as cq

result = (
cq.Workplane("XY")
.rect(30, 20)
.extrude(8)
.faces(">Z")
.workplane()
.ellipse(12, 8)
.cutBlind(-4)
.faces(">Z[-2]")
.workplane()
.cskHole(5.0, 3.0, 8.0)
)
cq.exporters.export(result, 'GT.stl')