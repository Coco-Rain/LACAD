import cadquery as cq

result = (
cq.Workplane("XY")
.rect(30, 20)
.extrude(8)
.faces(">Z")
.workplane()
.ellipse(12, 6)
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')