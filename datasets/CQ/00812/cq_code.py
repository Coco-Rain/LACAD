import cadquery as cq

result = (
cq.Workplane("XY")
.rect(30, 20)
.extrude(10)
.faces(">Z")
.workplane()
.ellipse(12, 8)
.extrude(5)
.faces(">Z")
.workplane()
.circle(3)
.hole(8)
)
result
cq.exporters.export(result, 'GT.stl')