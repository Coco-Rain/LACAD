import cadquery as cq

result = (
cq.Workplane("XY")
.rect(30, 20)
.extrude(5)
.faces(">Z")
.workplane()
.ellipse(10, 6)
.extrude(8)
)
cq.exporters.export(result, 'GT.stl')