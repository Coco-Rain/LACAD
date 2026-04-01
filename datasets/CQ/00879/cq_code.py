import cadquery as cq

result = (
cq.Workplane("XY")
.circle(5)
.extrude(10)
.faces(">Z")
.workplane()
.rect(6, 3)
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')