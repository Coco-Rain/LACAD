import cadquery as cq

result = (
cq.Workplane("XY")
.rect(50, 30)
.extrude(8)
.faces(">Z")
.workplane()
.ellipse(18, 10)
.extrude(6)
.faces(">Z")
.workplane()
.transformed(offset=(15, 0, 0))
.cboreHole(4.0, 8.0, 3.0, 5.0)
)
cq.exporters.export(result, 'GT.stl')