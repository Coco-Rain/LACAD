import cadquery as cq

result = (
cq.Workplane("XY")
.circle(5)
.extrude(3)
.faces(">Z")
.workplane()
.circle(3)
.extrude(2)
.last()
)
cq.exporters.export(result, 'GT.stl')